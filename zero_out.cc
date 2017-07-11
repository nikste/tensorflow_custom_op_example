#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"


//https://www.tensorflow.org/extend/adding_an_op#implement-the-kernel-for-the-op
//https://github.com/MatteoRagni/tf.ZeroOut.gpu/blob/master/zero_out.cc

using namespace tensorflow;


//REGISTER_OP("ZeroOut")
//  .Input("to_zero: int32")
//  .Output("zeroed: int32")
//  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
//    c->set_output(0, c->input(0));
//    return Status::OK();
//  });

REGISTER_OP("ZeroOut")
  .Input("to_zero: T")
  .Output("zeroed: T")
  .Attr("T: {int32, float}")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

// zero_out.cc
#define EIGEN_USE_THREADS
#include "zero_out.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct ZeroOutFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ZeroOutFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZeroOutOp<CPUDevice, T>);
//.TypeConstraint<T>("T")
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ZeroOut").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ZeroOutOp<GPUDevice, T>);
//.TypeConstraint<T>("T")
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

