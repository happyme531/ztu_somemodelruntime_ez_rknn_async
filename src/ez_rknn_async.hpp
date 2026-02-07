#pragma once

#include "rknn_api.h"
#include <any>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono> // 用于时间记录
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map> // 新增 std::map 用于顺序回调
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
// 包含 pthread 头文件以使用 pthread_setname_np
#include "rknn_dtype.hpp"
#include <pthread.h>

#if defined(__has_include)
#if __has_include("rknn_custom_op.h")
#include "rknn_custom_op.h"
#define ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP 1
#else
#define ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP 0
#endif
#else
#include "rknn_custom_op.h"
#define ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP 1
#endif

// 如果定义了 TRACY_ENABLE，则包含 Tracy 头文件
#ifdef TRACY_ENABLE
#include "tracy/Tracy.hpp"
#endif

#if __cplusplus < 201703L
#error                                                                         \
    "C++17 or a later version is required to compile this class. Please use a C++17 compatible compiler and enable C++17 features (e.g., -std=c++17)."
// 中文错误信息：
// #error "这个类要求使用 C++17 或更高版本的标准进行编译。请使用兼容 C++17
// 的编译器，并启用 C++17 特性（例如 -std=c++17）。"
#endif

// 如果未定义 ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT，默认使用
// dup_context（不禁用）
#ifndef ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT
#define ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT 0
#endif

namespace ztu {
namespace rk {
// 异步推理类，基于回调函数返回推理结果
// 不再需要 EnableTrace 模板参数
class AsyncEzRknn {
public:
#if ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP
  using GetCustomOpFunc = rknn_custom_op *(*)();
#else
  using GetCustomOpFunc = void *(*)();
#endif
  static constexpr const char *kCustomOpDisabledMsg =
      "Custom op support disabled (missing rknn_custom_op.h)";

  // 与 EzRknn 类似的输入数据排列方式
  enum class Layout {
    NCHW,
    ORIGINAL = NCHW,
    NHWC,
    ANY,
  };

  // 回调函数类型：参数为任务 id 和输出结果（每个输出为智能指针，指向拷贝后的
  // float 数组）
  using InferenceResult = std::vector<std::shared_ptr<float[]>>;
  using InferenceCallback =
      std::function<void(size_t taskId, InferenceResult outputs)>;

  // 动态输入视图（用于非模板接口，例如 Python 绑定）
  struct InputView {
    const void *data = nullptr;
    size_t bytes = 0;
    rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
  };

  // 构造函数
  // model_path：模型文件路径
  // layout：输入数据的布局
  // maxQueueSize：输入任务队列的最大个数（超过时不创建新任务）
  // threadsPerCore：每个 NPU 核心上创建的工作线程数
  // sequentialCallbacks：是否按顺序调用回调函数（默认 false，不保证顺序）
  // schedule：指定 NPU 核心的调度顺序 (例如 {0, 1, 2} 或 {0, 0, 1})
  AsyncEzRknn(const std::filesystem::path &model_path,
              Layout layout = Layout::ORIGINAL, size_t maxQueueSize = 3,
              int threadsPerCore = 2, bool sequentialCallbacks = true,
              std::vector<uint64_t> schedule = {0, 1,
                                                2}, // 默认使用 3 个核心轮询
              bool enablePacing = false)
      : model_path_(model_path), layout_(layout), maxQueueSize_(maxQueueSize),
        threadsPerCore_(threadsPerCore), stopFlag(false), taskCounter(0),
        sequentialCallbacks_(sequentialCallbacks), nextSequentialTask(0),
        schedule_(schedule), enablePacing_(enablePacing) {
    // 使用 Tracy 标记构造函数
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::Constructor");
    // 设置程序名称，方便在 Tracy 中识别
    TracySetProgramName("AsyncEzRknn");
#endif
    // Pacer 初始化
    if (enablePacing_) {
      std::set<uint64_t> used_cores(schedule.begin(), schedule.end());
      num_cores_ = used_cores.size();
      last_accepted_time_ = std::chrono::high_resolution_clock::now();
    }
    // 不再需要记录 traceStartTime
    init(); // 加载模型、创建初始上下文并查询输入输出属性，创建工作线程（每个线程拥有独立的
            // RKNN 上下文）
    // 启动专门用于调用回调函数的线程
    callbackThread = std::thread(&AsyncEzRknn::callbackThreadFunc, this);
  }

  ~AsyncEzRknn() {
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::Destructor");
#endif
    stopFlag = true;
    queueCond.notify_all();
    callbackCond.notify_all();
    for (auto &thread : workerThreads) {
      if (thread.joinable())
        thread.join();
    }
    if (callbackThread.joinable())
      callbackThread.join();
    // 不再需要 flushTraceEvents
    // 释放所有上下文
    for (auto ctx : contexts) {
      if (ctx != 0) {
        rknn_destroy(ctx);
      }
    }
#if ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP
    for (void *handle : so_handles_) {
      if (handle) {
        dlclose(handle);
      }
    }
    so_handles_.clear();
#endif
  }

#if ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP
  void load_custom_op(GetCustomOpFunc custom_op_func) {
    register_custom_op_for_all_contexts(custom_op_func);
  }

  void load_custom_op(const std::filesystem::path &so_path) {
    std::cout << "Loading custom op plugin: " << so_path.string() << std::endl;
    void *plugin_lib = dlopen(so_path.c_str(), RTLD_NOW);
    char *error = dlerror();
    if (error != NULL || plugin_lib == nullptr) {
      throw std::runtime_error("dlopen " + so_path.string() +
                               " fail: " + (error ? std::string(error) : ""));
    }

    GetCustomOpFunc custom_op_func =
        (GetCustomOpFunc)dlsym(plugin_lib, "get_rknn_custom_op");
    error = dlerror();
    if (error != NULL) {
      dlclose(plugin_lib);
      throw std::runtime_error("dlsym fail for get_rknn_custom_op: " +
                               std::string(error));
    }

    register_custom_op_for_all_contexts(custom_op_func);
    so_handles_.push_back(plugin_lib);
    std::cout << "Successfully loaded and registered " << so_path.string()
              << std::endl;
  }

  void load_custom_ops_from_default_path() {
    std::string plugin_dir;
#if defined(__ANDROID__)
#if defined(__aarch64__)
    plugin_dir = "/vendor/lib64/";
#else
    plugin_dir = "/vendor/lib/";
#endif
#elif defined(__linux__)
    plugin_dir = "/usr/lib/rknpu/op_plugins/";
#else
    std::cerr << "Warning: Default custom op path is not defined for this OS."
              << std::endl;
    return;
#endif

    if (!std::filesystem::exists(plugin_dir)) {
      std::cerr << "Warning: Default plugin directory does not exist: "
                << plugin_dir << std::endl;
      return;
    }

    const std::string prefix = "librkcst_";
    for (const auto &entry : std::filesystem::directory_iterator(plugin_dir)) {
      if (entry.is_regular_file()) {
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0 &&
            entry.path().extension() == ".so") {
          try {
            load_custom_op(entry.path());
          } catch (const std::runtime_error &e) {
            std::cerr << "Failed to load plugin " << filename << ": "
                      << e.what() << std::endl;
          }
        }
      }
    }
  }
#else
  void load_custom_op(GetCustomOpFunc) {
    throw std::runtime_error(kCustomOpDisabledMsg);
  }

  void load_custom_op(const std::filesystem::path &) {
    throw std::runtime_error(kCustomOpDisabledMsg);
  }

  void load_custom_ops_from_default_path() {
    std::cerr << "Warning: " << kCustomOpDisabledMsg << std::endl;
  }
#endif

  // 异步推理接口
  // 模板参数 Args 必须与模型的输入个数一致
  // 如果任务队列已满则直接返回 std::nullopt，否则返回任务 id
  template <typename... Args>
  std::optional<size_t> asyncInference(InferenceCallback callback,
                                       Args *...inputs) {
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::asyncInference");
#endif

    if (enablePacing_) {
      std::lock_guard<std::mutex> lock(pacer_mutex_);
      float avg_us = avg_proc_time_us_.load();
      if (avg_us > 0.0f) {
        auto interval_us =
            static_cast<long long>(avg_us / (num_cores_ > 0 ? num_cores_ : 1));
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              now - last_accepted_time_)
                              .count();
        if (elapsed_us < interval_us) {
          return std::nullopt; // Pacer says drop
        }
      }
      last_accepted_time_ = std::chrono::high_resolution_clock::now();
    }

    if (sizeof...(inputs) != input_attrs.size()) {
      throw std::runtime_error("Input count mismatch");
    }
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      if (taskQueue.size() >= maxQueueSize_) {
#ifdef TRACY_ENABLE
        // 记录任务队列满的事件
        TracyMessageL("Task queue full, rejecting new task");
#endif
        return std::nullopt; // 队列已满，不创建任务
      }
    }
    // 新增检查回调队列大小，超过阈值则拒绝提交新任务
    {
      std::lock_guard<std::mutex> lock(callbackMutex);
      if (sequentialCallbacks_) {
        if (pendingCallbacks.size() >= MAX_CALLBACK_QUEUE_SIZE) {
#ifdef TRACY_ENABLE
          TracyMessageL("Pending callback queue full, rejecting new task");
#endif
          return std::nullopt; // 回调队列已满，不创建任务
        }
      } else {
        if (callbackQueue.size() >= MAX_CALLBACK_QUEUE_SIZE) {
#ifdef TRACY_ENABLE
          TracyMessageL("Callback queue full, rejecting new task");
#endif
          return std::nullopt; // 回调队列已满，不创建任务
        }
      }
    }
    std::vector<std::vector<uint8_t>> inputCopies;
    std::vector<rknn_tensor_type> inputTypes; // 存储输入类型
    inputCopies.reserve(sizeof...(inputs));
    inputTypes.reserve(sizeof...(inputs));
    storeAllInputs(inputCopies, inputTypes,
                   inputs...); // 拷贝输入数据并存储类型

    size_t taskId = taskCounter++;
    size_t coreId = schedule_[taskId % schedule_.size()];
    Task task{taskId, std::move(inputCopies), std::move(inputTypes),
              std::move(callback), coreId};

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      taskQueue.push(std::move(task));
#ifdef TRACY_ENABLE
      // 绘制任务队列大小
      TracyPlot("Task Queue Size", (int64_t)taskQueue.size());
#endif
    }
    queueCond.notify_all();
    return taskId;
  }

  // 动态输入版本：输入数量与类型在运行时确定
  std::optional<size_t>
  asyncInferenceDyn(InferenceCallback callback,
                    const std::vector<InputView> &inputs) {
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::asyncInferenceDyn");
#endif

    if (enablePacing_) {
      std::lock_guard<std::mutex> lock(pacer_mutex_);
      float avg_us = avg_proc_time_us_.load();
      if (avg_us > 0.0f) {
        auto interval_us =
            static_cast<long long>(avg_us / (num_cores_ > 0 ? num_cores_ : 1));
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              now - last_accepted_time_)
                              .count();
        if (elapsed_us < interval_us) {
          return std::nullopt; // Pacer says drop
        }
      }
      last_accepted_time_ = std::chrono::high_resolution_clock::now();
    }

    if (inputs.size() != input_attrs.size()) {
      throw std::runtime_error("Input count mismatch");
    }
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      if (taskQueue.size() >= maxQueueSize_) {
#ifdef TRACY_ENABLE
        TracyMessageL("Task queue full, rejecting new task");
#endif
        return std::nullopt;
      }
    }
    {
      std::lock_guard<std::mutex> lock(callbackMutex);
      if (sequentialCallbacks_) {
        if (pendingCallbacks.size() >= MAX_CALLBACK_QUEUE_SIZE) {
#ifdef TRACY_ENABLE
          TracyMessageL("Pending callback queue full, rejecting new task");
#endif
          return std::nullopt;
        }
      } else {
        if (callbackQueue.size() >= MAX_CALLBACK_QUEUE_SIZE) {
#ifdef TRACY_ENABLE
          TracyMessageL("Callback queue full, rejecting new task");
#endif
          return std::nullopt;
        }
      }
    }

    std::vector<std::vector<uint8_t>> inputCopies;
    std::vector<rknn_tensor_type> inputTypes;
    inputCopies.reserve(inputs.size());
    inputTypes.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].data == nullptr) {
        throw std::runtime_error("Input data is null");
      }
      size_t expected_bytes =
          input_attrs[i].n_elems * rknnTypeSize(inputs[i].type);
      if (inputs[i].bytes != expected_bytes) {
        throw std::runtime_error(
            "Input size mismatch at index " + std::to_string(i) +
            ": expected " + std::to_string(expected_bytes) + " bytes, got " +
            std::to_string(inputs[i].bytes));
      }
      std::vector<uint8_t> buf(expected_bytes);
      std::memcpy(buf.data(), inputs[i].data, expected_bytes);
      inputCopies.push_back(std::move(buf));
      inputTypes.push_back(inputs[i].type);
    }

    size_t taskId = taskCounter++;
    size_t coreId = schedule_[taskId % schedule_.size()];
    Task task{taskId, std::move(inputCopies), std::move(inputTypes),
              std::move(callback), coreId};

    {
      std::lock_guard<std::mutex> lock(queueMutex);
      taskQueue.push(std::move(task));
#ifdef TRACY_ENABLE
      TracyPlot("Task Queue Size", (int64_t)taskQueue.size());
#endif
    }
    queueCond.notify_all();
    return taskId;
  }

  // 伪同步流水线推理器：单线程使用，每次调用提交一个新任务并在填满延迟后返回旧结果
  class PipelineRunner {
  public:
    using Result = InferenceResult;

    PipelineRunner(AsyncEzRknn &parent, size_t pipelineDepth)
        : parent_(parent), depth_(pipelineDepth) {
      if (depth_ == 0) {
        throw std::invalid_argument("Pipeline depth must be greater than 0");
      }
    }

    PipelineRunner(const PipelineRunner &) = delete;
    PipelineRunner &operator=(const PipelineRunner &) = delete;
    PipelineRunner(PipelineRunner &&) = default;
    PipelineRunner &operator=(PipelineRunner &&) = default;

    template <typename... Args>
    std::optional<Result> operator()(Args *...inputs) {
      enqueueTask(inputs...);
      if (pendingResults_.size() <= depth_) {
        return std::nullopt;
      }
      return finalizeResult();
    }

    template <typename... Args> auto inference(Args *...inputs) {
      return (*this)(inputs...);
    }

    template <typename UserData, typename... Args>
    std::optional<std::pair<std::decay_t<UserData>, Result>>
    inferenceWithData(UserData &&userData, Args *...inputs) {
      using StoredUserData = std::decay_t<UserData>;
      StoredUserData dataHolder(std::forward<UserData>(userData));
      enqueueTaskWithData(std::move(dataHolder), inputs...);
      if (pendingResults_.size() <= depth_) {
        return std::nullopt;
      }
      return finalizeResultWithData<StoredUserData>();
    }

    bool hasReadyResult() const { return pendingResults_.size() > depth_; }

  private:
    struct PendingItem {
      PendingItem(std::future<Result> &&futureIn,
                  std::optional<std::any> &&userDataIn)
          : future(std::move(futureIn)), userData(std::move(userDataIn)) {}
      PendingItem(const PendingItem &) = delete;
      PendingItem &operator=(const PendingItem &) = delete;
      PendingItem(PendingItem &&) = default;
      PendingItem &operator=(PendingItem &&) = default;

      std::future<Result> future;
      std::optional<std::any> userData;
    };

    template <typename... Args> void enqueueTask(Args *...inputs) {
      enqueueTaskImpl(std::nullopt, inputs...);
    }

    template <typename UserData, typename... Args>
    void enqueueTaskWithData(UserData &&userData, Args *...inputs) {
      std::optional<std::any> wrappedData;
      wrappedData.emplace(std::forward<UserData>(userData));
      enqueueTaskImpl(std::move(wrappedData), inputs...);
    }

    template <typename... Args>
    void enqueueTaskImpl(std::optional<std::any> userData, Args *...inputs) {
      auto promisePtr = std::make_shared<std::promise<Result>>();
      std::future<Result> fut = promisePtr->get_future();
      while (true) {
        auto submitted = parent_.asyncInference(
            [promisePtr](size_t, Result outputs) mutable {
              try {
                promisePtr->set_value(std::move(outputs));
              } catch (...) {
                try {
                  promisePtr->set_exception(std::current_exception());
                } catch (...) {
                }
              }
            },
            inputs...);
        if (submitted.has_value()) {
          std::optional<std::any> storedUserData;
          if (userData.has_value()) {
            storedUserData.emplace(std::move(*userData));
          }
          pendingResults_.emplace(std::move(fut), std::move(storedUserData));
          break;
        }
        // std::this_thread::yield(); // 等队列有空位
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    Result finalizeResult() {
      PendingItem item = std::move(pendingResults_.front());
      pendingResults_.pop();
      if (item.userData.has_value()) {
        throw std::logic_error(
            "PipelineRunner: result with user data retrieved without using "
            "inferenceWithData.");
      }
      return item.future.get();
    }

    template <typename UserData>
    std::pair<UserData, Result> finalizeResultWithData() {
      PendingItem item = std::move(pendingResults_.front());
      pendingResults_.pop();
      if (!item.userData.has_value()) {
        throw std::logic_error(
            "PipelineRunner: expected user data is missing for this result.");
      }
      auto extractData = [&]() -> UserData {
        try {
          return std::any_cast<UserData>(std::move(*item.userData));
        } catch (const std::bad_any_cast &) {
          throw std::logic_error(
              "PipelineRunner: stored user data type does not match requested "
              "type.");
        }
      };
      UserData data = extractData();
      Result outputs = item.future.get();
      return {std::move(data), std::move(outputs)};
    }

    AsyncEzRknn &parent_;
    size_t depth_;
    std::queue<PendingItem> pendingResults_;
  };

  rknn_input_output_num io_num;
  std::vector<rknn_tensor_attr> input_attrs;
  std::vector<rknn_tensor_attr> output_attrs;
  std::vector<rknn_context> contexts; // 每个工作线程拥有独立的上下文
  const std::optional<std::string> &sdk_version_warning() const {
    return sdk_version_warning_;
  }

private:
  // Pacing / Smoothing members
  bool enablePacing_ = false;
  std::atomic<float> avg_proc_time_us_{0.0f};
  std::chrono::high_resolution_clock::time_point last_accepted_time_;
  std::mutex pacer_mutex_;
  size_t num_cores_ = 1;

  // 内部任务结构：存储任务 id、输入数据拷贝和回调函数
  struct Task {
    size_t taskId;
    std::vector<std::vector<uint8_t>> inputs; // 每个输入数据拷贝
    std::vector<rknn_tensor_type> inputTypes; // 每个输入的类型
    InferenceCallback callback;
    size_t coreId;
  };

  // 成员变量
  std::filesystem::path model_path_;
  Layout layout_;
  size_t maxQueueSize_;
  int threadsPerCore_;
  std::vector<uint64_t> schedule_;

  // RKNN 相关变量
  rknn_context initial_ctx = 0;
  std::map<int, std::timed_mutex> coreMutexes_; // 修改: 使用 std::timed_mutex

  // 线程池及任务队列
  std::vector<std::thread> workerThreads;
  std::queue<Task> taskQueue;
  std::mutex queueMutex;
  std::condition_variable queueCond;
  std::atomic<bool> stopFlag;
  std::atomic<size_t> taskCounter;

  // 专门调用回调函数的队列及线程
  std::mutex callbackMutex;
  std::condition_variable callbackCond;
  std::queue<std::function<void()>> callbackQueue;
  std::thread callbackThread;
  static constexpr size_t MAX_CALLBACK_QUEUE_SIZE = 8;
  // 新增顺序回调相关成员
  bool sequentialCallbacks_ = false;
  size_t nextSequentialTask = 0;
  std::map<size_t, std::function<void()>> pendingCallbacks;
#if ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP
  std::vector<void *> so_handles_;
#endif
  std::optional<std::string> sdk_version_warning_;

  // 修改后的 enqueueCallback 函数，直接添加回调，不阻塞等待
  void enqueueCallback(size_t taskId, std::function<void()> cb) {
#ifdef TRACY_ENABLE
    // 使用 Tracy 标记，但注意避免锁内部分配内存，如果可能的话
    // 这里仅作标记，不传递复杂参数
    ZoneScopedN("AsyncEzRknn::enqueueCallback");
#endif
    std::lock_guard<std::mutex> lock(callbackMutex);
    if (sequentialCallbacks_) {
      pendingCallbacks[taskId] = std::move(cb);
#ifdef TRACY_ENABLE
      // 绘制顺序回调队列大小
      TracyPlot("Pending Callbacks Size", (int64_t)pendingCallbacks.size());
#endif
    } else {
      callbackQueue.push(std::move(cb));
#ifdef TRACY_ENABLE
      // 绘制非顺序回调队列大小
      TracyPlot("Callback Queue Size", (int64_t)callbackQueue.size());
#endif
    }
    callbackCond.notify_all();
  }

  static bool parse_version_triplet(const char *version_text,
                                    std::array<int, 3> &out) {
    out = {0, 0, 0};
    if (version_text == nullptr) {
      return false;
    }
    int idx = 0;
    int value = 0;
    bool reading_number = false;
    bool has_any_number = false;
    for (const char *p = version_text; *p != '\0' && idx < 3; ++p) {
      const unsigned char ch = static_cast<unsigned char>(*p);
      if (std::isdigit(ch)) {
        reading_number = true;
        has_any_number = true;
        value = value * 10 + static_cast<int>(ch - '0');
      } else if (reading_number) {
        out[idx++] = value;
        value = 0;
        reading_number = false;
      }
    }
    if (reading_number && idx < 3) {
      out[idx++] = value;
    }
    return has_any_number && idx > 0;
  }

  static bool is_version_less_than(const std::array<int, 3> &lhs, int maj,
                                   int min, int patch) {
    const std::array<int, 3> rhs = {maj, min, patch};
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i] < rhs[i]) {
        return true;
      }
      if (lhs[i] > rhs[i]) {
        return false;
      }
    }
    return false;
  }

  void warn_if_rknn_sdk_too_old() {
    sdk_version_warning_.reset();
    rknn_sdk_version sdk_version;
    std::memset(&sdk_version, 0, sizeof(sdk_version));
    const int ret = rknn_query(initial_ctx, RKNN_QUERY_SDK_VERSION,
                               &sdk_version, sizeof(sdk_version));
    if (ret < 0) {
      return;
    }

    std::array<int, 3> parsed = {0, 0, 0};
    if (!parse_version_triplet(sdk_version.api_version, parsed)) {
      return;
    }

    if (is_version_less_than(parsed, 2, 3, 2)) {
      sdk_version_warning_ = "RKNN API version " +
                             std::string(sdk_version.api_version) +
                             " is likely outdated; behavior may be "
                             "unstable.";
    }
  }

  // 初始化：加载模型文件、初始化 RKNN 上下文并查询 IO 属性
  void init() {
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::init");
#endif
    std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open model file: " +
                               model_path_.string());
    }
    size_t model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    if (!file.read(reinterpret_cast<char *>(model_data.data()), model_size)) {
      throw std::runtime_error("Failed to read model file");
    }
    int ret =
        rknn_init(&initial_ctx, model_data.data(), model_size, 0, nullptr);
    if (ret < 0) {
      throw std::runtime_error("rknn_init failed with error code: " +
                               std::to_string(ret));
    }
    warn_if_rknn_sdk_too_old();
    ret =
        rknn_query(initial_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
      throw std::runtime_error(
          "rknn_query for IO num failed with error code: " +
          std::to_string(ret));
    }
    input_attrs.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; i++) {
      input_attrs[i].index = i;
      ret = rknn_query(initial_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i],
                       sizeof(rknn_tensor_attr));
      if (ret < 0) {
        throw std::runtime_error("Failed to query input attr for index " +
                                 std::to_string(i));
      }
      if (input_attrs[i].n_dims == 4 && layout_ == Layout::ORIGINAL) {
        auto n = input_attrs[i].dims[0];
        auto h = input_attrs[i].dims[1];
        auto w = input_attrs[i].dims[2];
        auto c = input_attrs[i].dims[3];
        input_attrs[i].dims[0] = n;
        input_attrs[i].dims[1] = c;
        input_attrs[i].dims[2] = h;
        input_attrs[i].dims[3] = w;
      }
    }
    output_attrs.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
      output_attrs[i].index = i;
      ret = rknn_query(initial_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i],
                       sizeof(rknn_tensor_attr));
      if (ret < 0) {
        throw std::runtime_error("Failed to query output attr for index " +
                                 std::to_string(i));
      }
    }

    // 根据 core_mask_ 和每个 NPU 核的线程数创建工作线程，并复制上下文
#ifdef TRACY_ENABLE
    ZoneScopedN("AsyncEzRknn::startWorkerThreads");
#endif
    std::set<uint64_t> used_cores(schedule_.begin(), schedule_.end());
    std::vector<uint64_t> used_cores_vec(used_cores.begin(), used_cores.end());
    if (used_cores_vec.empty()) {
      throw std::runtime_error("No NPU core selected");
    }

    // 初始化每个核心的互斥锁
    for (uint64_t core : used_cores_vec) {
      coreMutexes_[static_cast<int>(core)]; // 为每个使用的核心创建一个互斥锁
    }

    int totalThreads = threadsPerCore_ * used_cores.size();
    contexts.resize(totalThreads);
    contexts[0] = initial_ctx;
    for (int i = 1; i < totalThreads; i++) {
#if ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT
      // 禁用 dup_context，使用 init 初始化每个 context
      int ret =
          rknn_init(&contexts[i], model_data.data(), model_size, 0, nullptr);
      if (ret < 0) {
        throw std::runtime_error("rknn_init failed with error code: " +
                                 std::to_string(ret));
      }
#else
      // 使用 dup_context 复制初始 context
      int ret = rknn_dup_context(&initial_ctx, &contexts[i]);
      if (ret < 0) {
        throw std::runtime_error("rknn_dup_context failed with error code: " +
                                 std::to_string(ret));
      }
#endif
    }
    for (int i = 0; i < totalThreads; i++) {
      int core = used_cores_vec[i % used_cores_vec.size()];
      int ret = rknn_set_core_mask(contexts[i],
                                   static_cast<rknn_core_mask>(1 << core));
      if (ret < 0) {
        throw std::runtime_error("rknn_set_core_mask failed for thread " +
                                 std::to_string(i) +
                                 " with error code: " + std::to_string(ret));
      }
      workerThreads.emplace_back(&AsyncEzRknn::workerThreadFunc, this, i, core);
      // 为每个工作线程命名 (使用 pthread_setname_np)
      std::string threadName =
          "EzRknn_W" + std::to_string(i) + "_C" + std::to_string(core);
      // 注意：pthread_setname_np 对线程名称长度有限制（通常是15或16个字符）
      // 我们需要截断或缩写名称以确保符合要求
      threadName.resize(15); // 截断为15个字符
      pthread_setname_np(workerThreads.back().native_handle(),
                         threadName.c_str());
    }
  }

  // 回调线程函数：调用回调函数，支持顺序模式和非顺序模式
  void callbackThreadFunc() {
    // 给回调线程命名 (使用 pthread_setname_np)
    // 在线程函数内部调用时，使用 pthread_self() 获取当前线程句柄
    pthread_setname_np(pthread_self(), "EzRknn_Callback");
#ifdef TRACY_ENABLE
    ZoneScopedN("CallbackThreadFunc");
#endif
    if (sequentialCallbacks_) {
      while (!stopFlag) {
        std::function<void()> callbackToExec;
        {
          std::unique_lock<std::mutex> lock(callbackMutex);
#ifdef TRACY_ENABLE
          // 标记等待回调
          ZoneScopedN("CallbackWaitSequential");
#endif
          callbackCond.wait(lock, [this] {
            return stopFlag || (pendingCallbacks.find(nextSequentialTask) !=
                                pendingCallbacks.end());
          });
          if (stopFlag) {
            break;
          }
          auto it = pendingCallbacks.find(nextSequentialTask);
          if (it != pendingCallbacks.end()) {
            callbackToExec = std::move(it->second);
            pendingCallbacks.erase(it);
#ifdef TRACY_ENABLE
            // 绘制顺序回调队列大小
            TracyPlot("Pending Callbacks Size",
                      (int64_t)pendingCallbacks.size());
#endif
            nextSequentialTask++;
            callbackCond
                .notify_all(); // 可能需要唤醒等待特定任务ID的线程（如果以后有这种需求）
          }
        }
        if (callbackToExec) {
#ifdef TRACY_ENABLE
          // 标记执行回调
          ZoneScopedN("ExecuteSequentialCallback");
          // 可以添加任务 ID 作为文本信息
          // std::string msg = "TaskID: " + std::to_string(nextSequentialTask -
          // 1); ZoneText(msg.c_str(), msg.length()); // 注意：ZoneText
          // 可能有性能开销
#endif
          callbackToExec();
        }
      }
      // 处理停止后剩余的回调
      while (true) {
        std::function<void()> remainingCallback;
        {
          std::lock_guard<std::mutex> lock(callbackMutex);
          auto it = pendingCallbacks.find(nextSequentialTask);
          if (it == pendingCallbacks.end())
            break;
          remainingCallback = std::move(it->second);
          pendingCallbacks.erase(it);
#ifdef TRACY_ENABLE
          TracyPlot("Pending Callbacks Size", (int64_t)pendingCallbacks.size());
#endif
          nextSequentialTask++;
        }
        if (remainingCallback) {
#ifdef TRACY_ENABLE
          ZoneScopedN("ExecuteRemainingSequentialCallback");
#endif
          remainingCallback();
        }
      }
    } else { // 非顺序模式
      while (!stopFlag) {
        std::function<void()> cbTask;
        {
          std::unique_lock<std::mutex> lock(callbackMutex);
#ifdef TRACY_ENABLE
          // 标记等待回调
          ZoneScopedN("CallbackWaitNonSequential");
#endif
          callbackCond.wait(
              lock, [this] { return !callbackQueue.empty() || stopFlag; });
          if (!callbackQueue.empty()) {
            cbTask = std::move(callbackQueue.front());
            callbackQueue.pop();
#ifdef TRACY_ENABLE
            // 绘制非顺序回调队列大小
            TracyPlot("Callback Queue Size", (int64_t)callbackQueue.size());
#endif
            callbackCond
                .notify_all(); // 可能需要唤醒等待队列空间的线程（如果以后有这种需求）
          } else if (stopFlag) {
            break;
          }
        }
        if (cbTask) {
#ifdef TRACY_ENABLE
          // 标记执行回调
          ZoneScopedN("ExecuteNonSequentialCallback");
#endif
          cbTask();
        }
      }
      // 处理停止后剩余的回调
      while (true) {
        std::function<void()> remainingTask;
        {
          std::lock_guard<std::mutex> lock(callbackMutex);
          if (callbackQueue.empty())
            break;
          remainingTask = std::move(callbackQueue.front());
          callbackQueue.pop();
#ifdef TRACY_ENABLE
          TracyPlot("Callback Queue Size", (int64_t)callbackQueue.size());
#endif
        }
        if (remainingTask) {
#ifdef TRACY_ENABLE
          ZoneScopedN("ExecuteRemainingNonSequentialCallback");
#endif
          remainingTask();
        }
      }
    }
  }

  // 工作线程函数：从任务队列中取任务，并推理和回调
  void workerThreadFunc(int threadId, int coreId) {
    // 线程名称已在 startWorkerThreads 中设置
#ifdef TRACY_ENABLE
    // 标记整个工作线程的生命周期
    ZoneScopedN("WorkerThreadFunc");
#endif
    while (!stopFlag) {
      Task task;
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        {
#ifdef TRACY_ENABLE
          // 标记等待任务
          ZoneScopedN("WorkerWaitTask");
          // 可以添加线程 ID 和核心 ID 作为文本信息
          // std::string waitMsg = "Thread: " + std::to_string(threadId) + ",
          // Core: " + std::to_string(coreId); ZoneText(waitMsg.c_str(),
          // waitMsg.length());
#endif
          queueCond.wait(lock, [this, coreId] {
            return (!taskQueue.empty() && taskQueue.front().coreId == coreId) ||
                   stopFlag;
          });
        }
        if (stopFlag && taskQueue.empty()) {
          break;
        }
        task = std::move(taskQueue.front());
        taskQueue.pop();
        queueCond.notify_all(); // 让其他线程重新检查条件
#ifdef TRACY_ENABLE
        // 绘制任务队列大小
        TracyPlot("Task Queue Size", (int64_t)taskQueue.size());
#endif
      }
      {
#ifdef TRACY_ENABLE
        // 标记整个任务处理过程
        std::string taskArgs = "taskId:" + std::to_string(task.taskId);
        ZoneScopedNC("TaskProcessing",
                     tracy::Color::BlueViolet); // 使用不同颜色区分
        ZoneText(taskArgs.c_str(), taskArgs.length());
#endif
        rknn_context ctx = contexts[threadId];

        // 1. 构造并准备输入数据
        std::vector<rknn_input> rknnInputs;
        {
#ifdef TRACY_ENABLE
          ZoneScopedN("PrepareInputs");
#endif
          rknnInputs.reserve(input_attrs.size());
          for (size_t i = 0; i < input_attrs.size(); ++i) {
            rknn_input in = {};
            in.index = static_cast<uint32_t>(i);
            in.type = task.inputTypes[i];
            if (input_attrs[i].n_dims == 4) {
              in.fmt = (layout_ == Layout::NCHW ? RKNN_TENSOR_NCHW
                                                : RKNN_TENSOR_NHWC);
            } else {
              in.fmt = RKNN_TENSOR_UNDEFINED;
            }
            in.buf = task.inputs[i].data();
            in.size = task.inputs[i].size();
            rknnInputs.push_back(in);
          }
        }

        // 2. 设置输入数据
        int ret;
        {
#ifdef TRACY_ENABLE
          ZoneScopedN("SetInputs");
#endif
          ret = rknn_inputs_set(ctx, static_cast<uint32_t>(rknnInputs.size()),
                                rknnInputs.data());
        }
        if (ret < 0) {
#ifdef TRACY_ENABLE
          // 记录错误信息
          std::string errorMsg =
              "rknn_inputs_set Error: " + std::to_string(ret);
          TracyMessageLC(errorMsg.c_str(), tracy::Color::Red);
#endif
          enqueueCallback(task.taskId, [cb = task.callback,
                                        id = task.taskId]() { cb(id, {}); });
          continue;
        }

        // 3. 执行推理 (尝试加锁 5ms)
        int ret_run;
        {
          // 获取当前任务对应核心的 timed_mutex
          std::timed_mutex &currentCoreMutex = coreMutexes_.at(task.coreId);
          std::unique_lock<std::timed_mutex> lock(
              currentCoreMutex, std::defer_lock); // 准备 unique_lock
#ifdef TRACY_ENABLE
          ZoneScopedN("AttemptLockAndRunInference"); // 更新 Tracy Zone 名称
                                                     // 添加核心 ID 信息 (可选)
          // std::string lockMsg = "Attempting lock on core: " +
          // std::to_string(task.coreId); ZoneText(lockMsg.c_str(),
          // lockMsg.length());
#endif
          // // // 尝试获取锁
          // lock.try_lock_for(std::chrono::milliseconds(
          //     static_cast<int>(avg_proc_time_us_.load())));
          // 成功获取锁
          {
#ifdef TRACY_ENABLE
            // 标记持有锁并执行推理
            ZoneScopedN("RunInference (Locked)");
#endif
            // 在锁的作用域内执行 rknn_run
            auto inference_start_time =
                std::chrono::high_resolution_clock::now();
            ret_run = rknn_run(ctx, nullptr);
            auto inference_end_time = std::chrono::high_resolution_clock::now();
            if (ret_run >= 0 && enablePacing_) {
              auto proc_time_us =
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      inference_end_time - inference_start_time)
                      .count();
              float current_avg = avg_proc_time_us_.load();
              if (current_avg == 0.0f) {
                avg_proc_time_us_.store(static_cast<float>(proc_time_us));
              } else {
                avg_proc_time_us_.store(0.95f * current_avg +
                                        0.05f *
                                            static_cast<float>(proc_time_us));
              }
#ifdef TRACY_ENABLE
              TracyPlot("Current Proc Time (us)", proc_time_us);
#endif
            }
            // unique_lock 会在作用域结束时自动解锁
          }
        } // 结束尝试加锁和推理的作用域

        if (ret_run < 0) {
#ifdef TRACY_ENABLE
          std::string errorMsg = "rknn_run Error: " + std::to_string(ret_run);
          TracyMessageLC(errorMsg.c_str(), tracy::Color::Red);
#endif
          enqueueCallback(task.taskId, [cb = task.callback,
                                        id = task.taskId]() { cb(id, {}); });
          continue;
        }

        // 4. 拷贝输出数据（使用 prealloc 模式避免拷贝）
        {
#ifdef TRACY_ENABLE
          ZoneScopedN("CopyOutputs");
#endif
          std::vector<rknn_output> outputs(io_num.n_output);
          std::vector<std::shared_ptr<float[]>> outputResults(io_num.n_output);
          for (uint32_t i = 0; i < io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = true;
            outputs[i].is_prealloc = true; // 保持 prealloc
            uint32_t num_elems = output_attrs[i].n_elems;
            uint32_t buf_size = num_elems * sizeof(float);
            // 注意：prealloc 模式下，需要确保分配的内存足够大
            float *buf =
                new (std::nothrow) float[num_elems]; // 使用 nothrow 避免异常
            if (!buf) {
              // 内存分配失败处理
#ifdef TRACY_ENABLE
              std::string errorMsg =
                  "Failed to allocate output buffer for output " +
                  std::to_string(i);
              TracyMessageLC(errorMsg.c_str(), tracy::Color::Red);
#endif
              // 清理已分配的内存
              for (uint32_t j = 0; j < i; ++j) {
                delete[] reinterpret_cast<float *>(outputs[j].buf);
              }
              ret = -1; // 标记错误
              break;    // 跳出循环
            }
            outputs[i].buf = reinterpret_cast<void *>(buf);
            outputs[i].size = buf_size;
            // 使用自定义删除器管理 new[] 分配的内存
            outputResults[i] =
                std::shared_ptr<float[]>(buf, [](float *p) { delete[] p; });
          }

          // 检查内部分配是否失败
          if (ret < 0) {
            enqueueCallback(task.taskId, [cb = task.callback,
                                          id = task.taskId]() { cb(id, {}); });
            continue; // 继续下一个任务
          }

          ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr);

          if (ret < 0) {
#ifdef TRACY_ENABLE
            std::string errorMsg =
                "rknn_outputs_get Error: " + std::to_string(ret);
            TracyMessageLC(errorMsg.c_str(), tracy::Color::Red);
#endif
            // 即使 get 失败，之前分配的内存也需要通过智能指针自动释放
            // rknn_outputs_release 在 prealloc
            // 模式下通常不需要，但为了安全可以保留（它应该能处理 prealloc）
            rknn_outputs_release(ctx, io_num.n_output, outputs.data());
            enqueueCallback(task.taskId, [cb = task.callback,
                                          id = task.taskId]() { cb(id, {}); });
            continue;
          }
          // 在 prealloc 模式下，RKNN 不会接管内存，需要我们自己管理
          // 但 rknn_outputs_release 仍然需要调用以释放 RKNN
          // 内部的一些资源（如果有的话）
          rknn_outputs_release(ctx, io_num.n_output, outputs.data());

          // 输出数据已经通过 outputResults 的智能指针管理，直接移交即可
          enqueueCallback(task.taskId,
                          [cb = task.callback, id = task.taskId,
                           outs = std::move(outputResults)]() mutable {
#ifdef TRACY_ENABLE
                            // 标记回调入队后的 lambda 执行
                            ZoneScopedN("CallbackLambdaExecution");
#endif
                            cb(id, std::move(outs));
                          });
        }
      } // 结束 TaskProcessing Zone
    } // 结束 while 循环
  } // 结束 WorkerThreadFunc Zone

#if ZTU_EZRKNN_ASYNC_HAS_CUSTOM_OP
  void register_custom_op_for_all_contexts(GetCustomOpFunc custom_op_func) {
    if (custom_op_func == nullptr) {
      throw std::invalid_argument("custom_op_func is null");
    }

    // 检查是否需要警告：使用 dup_context 且总线程数大于 1
#if !ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT
    if (contexts.size() > 1) {
      std::cerr
          << "WARNING: Registering custom ops with dup_context and "
             "multiple threads (>1) may cause issues due to RKNN internal "
             "bugs. Consider defining ZTU_EZRKNN_ASYNC_DISABLE_DUP_CONTEXT "
             "to use rknn_init instead."
          << std::endl;
    }
#endif

    bool registered = false;
    for (auto ctx : contexts) {
      register_custom_op_for_context(ctx, custom_op_func);
      registered = true;
    }
    if (!registered && initial_ctx != 0) {
      register_custom_op_for_context(initial_ctx, custom_op_func);
      registered = true;
    }
    if (!registered) {
      throw std::runtime_error(
          "No RKNN context available for custom op registration");
    }
  }

  void register_custom_op_for_context(rknn_context ctx,
                                      GetCustomOpFunc custom_op_func) {
    if (ctx == 0) {
      throw std::runtime_error("RKNN context is null");
    }
    rknn_custom_op *user_op = custom_op_func();
    if (user_op == nullptr) {
      throw std::runtime_error("get_rknn_custom_op returned null");
    }
    int ret = rknn_register_custom_ops(ctx, user_op, 1);
    if (ret < 0) {
      throw std::runtime_error("rknn_register_custom_ops fail! ret = " +
                               std::to_string(ret));
    }
  }
#endif

  // 修改 storeInput 方法，添加类型信息
  template <typename T>
  void storeInput(std::vector<std::vector<uint8_t>> &storage,
                  std::vector<rknn_tensor_type> &types, size_t &index,
                  T *input) {
    size_t expectedSize = input_attrs[index].n_elems * sizeof(T);
    std::vector<uint8_t> buf(expectedSize);
    std::memcpy(buf.data(), input, expectedSize);
    storage.push_back(std::move(buf));
    types.push_back(RknnDtype<T>::type);
    index++;
  }

  template <typename... Args>
  void storeAllInputs(std::vector<std::vector<uint8_t>> &storage,
                      std::vector<rknn_tensor_type> &types, Args *...inputs) {
    size_t index = 0;
    (storeInput(storage, types, index, inputs), ...);
  }

  static size_t rknnTypeSize(rknn_tensor_type type) {
    switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return sizeof(float);
    case RKNN_TENSOR_FLOAT16:
      return sizeof(uint16_t);
    case RKNN_TENSOR_INT8:
      return sizeof(int8_t);
    case RKNN_TENSOR_UINT8:
      return sizeof(uint8_t);
    case RKNN_TENSOR_INT16:
      return sizeof(int16_t);
    case RKNN_TENSOR_UINT16:
      return sizeof(uint16_t);
    case RKNN_TENSOR_INT32:
      return sizeof(int32_t);
    case RKNN_TENSOR_UINT32:
      return sizeof(uint32_t);
    case RKNN_TENSOR_INT64:
      return sizeof(int64_t);
    case RKNN_TENSOR_BOOL:
      return sizeof(bool);
    default:
      throw std::runtime_error("Unsupported rknn tensor type");
    }
  }
};

} // namespace rk
} // namespace ztu
