#include "rknn_api.h"
#include "rknn_custom_op.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

class Rknpu2Runtime {
public:
  ~Rknpu2Runtime() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }

  void *symbol(const char *name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ensure_loaded())
      return nullptr;
    dlerror();
    void *result = dlsym(handle_, name);
    if (result == nullptr) {
      const char *error = dlerror();
      report_once(std::string("missing symbol ") + name + " in " +
                  loaded_path_ + (error ? std::string(": ") + error : ""));
    }
    return result;
  }

private:
  bool ensure_loaded() {
    if (handle_ != nullptr)
      return true;

    std::vector<std::string> candidates;
    if (const char *override_path =
            std::getenv("ZTU_RKNPU2_RUNTIME_LIBRARY")) {
      if (*override_path != '\0')
        candidates.emplace_back(override_path);
    }
    if (candidates.empty()) {
      candidates.emplace_back("librknnrt.so");
      candidates.emplace_back("librknnrt.so.2");
    }

    std::string errors;
    for (const auto &candidate : candidates) {
      dlerror();
      void *handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr) {
        handle_ = handle;
        loaded_path_ = candidate;
        return true;
      }
      const char *error = dlerror();
      if (error != nullptr) {
        if (!errors.empty())
          errors += "; ";
        errors += candidate + ": " + error;
      }
    }
    report_once(
        "unable to load the RKNPU2 runtime. Set "
        "ZTU_RKNPU2_RUNTIME_LIBRARY to librknnrt.so. Tried: " + errors);
    return false;
  }

  void report_once(const std::string &message) {
    if (reported_)
      return;
    reported_ = true;
    std::cerr << "[ztu_somemodelruntime_rknn] " << message << std::endl;
  }

  std::mutex mutex_;
  void *handle_ = nullptr;
  std::string loaded_path_;
  bool reported_ = false;
};

Rknpu2Runtime &runtime() {
  static Rknpu2Runtime instance;
  return instance;
}

template <typename Fn> Fn resolve(const char *name) {
  return reinterpret_cast<Fn>(runtime().symbol(name));
}

} // namespace

extern "C" {

int rknn_init(rknn_context *context, void *model, uint32_t size, uint32_t flag,
              rknn_init_extend *extend) {
  using Fn = int (*)(rknn_context *, void *, uint32_t, uint32_t,
                    rknn_init_extend *);
  auto fn = resolve<Fn>("rknn_init");
  return fn ? fn(context, model, size, flag, extend) : RKNN_ERR_FAIL;
}

int rknn_dup_context(rknn_context *context_in, rknn_context *context_out) {
  using Fn = int (*)(rknn_context *, rknn_context *);
  auto fn = resolve<Fn>("rknn_dup_context");
  return fn ? fn(context_in, context_out) : RKNN_ERR_FAIL;
}

int rknn_destroy(rknn_context context) {
  using Fn = int (*)(rknn_context);
  auto fn = resolve<Fn>("rknn_destroy");
  return fn ? fn(context) : RKNN_ERR_FAIL;
}

int rknn_query(rknn_context context, rknn_query_cmd cmd, void *info,
               uint32_t size) {
  using Fn = int (*)(rknn_context, rknn_query_cmd, void *, uint32_t);
  auto fn = resolve<Fn>("rknn_query");
  return fn ? fn(context, cmd, info, size) : RKNN_ERR_FAIL;
}

int rknn_inputs_set(rknn_context context, uint32_t n_inputs,
                    rknn_input inputs[]) {
  using Fn = int (*)(rknn_context, uint32_t, rknn_input *);
  auto fn = resolve<Fn>("rknn_inputs_set");
  return fn ? fn(context, n_inputs, inputs) : RKNN_ERR_FAIL;
}

int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask) {
  using Fn = int (*)(rknn_context, rknn_core_mask);
  auto fn = resolve<Fn>("rknn_set_core_mask");
  return fn ? fn(context, core_mask) : RKNN_ERR_FAIL;
}

int rknn_run(rknn_context context, rknn_run_extend *extend) {
  using Fn = int (*)(rknn_context, rknn_run_extend *);
  auto fn = resolve<Fn>("rknn_run");
  return fn ? fn(context, extend) : RKNN_ERR_FAIL;
}

int rknn_outputs_get(rknn_context context, uint32_t n_outputs,
                     rknn_output outputs[], rknn_output_extend *extend) {
  using Fn = int (*)(rknn_context, uint32_t, rknn_output *,
                    rknn_output_extend *);
  auto fn = resolve<Fn>("rknn_outputs_get");
  return fn ? fn(context, n_outputs, outputs, extend) : RKNN_ERR_FAIL;
}

int rknn_outputs_release(rknn_context context, uint32_t n_outputs,
                         rknn_output outputs[]) {
  using Fn = int (*)(rknn_context, uint32_t, rknn_output *);
  auto fn = resolve<Fn>("rknn_outputs_release");
  return fn ? fn(context, n_outputs, outputs) : RKNN_ERR_FAIL;
}

rknn_tensor_mem *rknn_create_mem_from_fd(rknn_context context, int32_t fd,
                                         void *virt_addr, uint32_t size,
                                         int32_t offset) {
  using Fn = rknn_tensor_mem *(*)(rknn_context, int32_t, void *, uint32_t,
                                 int32_t);
  auto fn = resolve<Fn>("rknn_create_mem_from_fd");
  return fn ? fn(context, fd, virt_addr, size, offset) : nullptr;
}

rknn_tensor_mem *rknn_create_mem(rknn_context context, uint32_t size) {
  using Fn = rknn_tensor_mem *(*)(rknn_context, uint32_t);
  auto fn = resolve<Fn>("rknn_create_mem");
  return fn ? fn(context, size) : nullptr;
}

int rknn_destroy_mem(rknn_context context, rknn_tensor_mem *mem) {
  using Fn = int (*)(rknn_context, rknn_tensor_mem *);
  auto fn = resolve<Fn>("rknn_destroy_mem");
  return fn ? fn(context, mem) : RKNN_ERR_FAIL;
}

int rknn_set_io_mem(rknn_context context, rknn_tensor_mem *mem,
                    rknn_tensor_attr *attr) {
  using Fn = int (*)(rknn_context, rknn_tensor_mem *, rknn_tensor_attr *);
  auto fn = resolve<Fn>("rknn_set_io_mem");
  return fn ? fn(context, mem, attr) : RKNN_ERR_FAIL;
}

int rknn_set_input_shapes(rknn_context context, uint32_t n_inputs,
                          rknn_tensor_attr attrs[]) {
  using Fn = int (*)(rknn_context, uint32_t, rknn_tensor_attr *);
  auto fn = resolve<Fn>("rknn_set_input_shapes");
  return fn ? fn(context, n_inputs, attrs) : RKNN_ERR_FAIL;
}

int rknn_mem_sync(rknn_context context, rknn_tensor_mem *mem,
                  rknn_mem_sync_mode mode) {
  using Fn = int (*)(rknn_context, rknn_tensor_mem *, rknn_mem_sync_mode);
  auto fn = resolve<Fn>("rknn_mem_sync");
  return fn ? fn(context, mem, mode) : RKNN_ERR_FAIL;
}

int rknn_register_custom_ops(rknn_context context, rknn_custom_op *ops,
                             uint32_t custom_op_num) {
  using Fn = int (*)(rknn_context, rknn_custom_op *, uint32_t);
  auto fn = resolve<Fn>("rknn_register_custom_ops");
  return fn ? fn(context, ops, custom_op_num) : RKNN_ERR_FAIL;
}

} // extern "C"
