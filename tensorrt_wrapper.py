"""
Simple and robust TensorRT inference wrapper.
"""

import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTModel:
    """
    TensorRT inference wrapper that handles memory management properly.
    """

    def __init__(self, engine_path, max_batch_size=4):
        """
        Initialize TensorRT model.

        Args:
            engine_path: Path to .engine file
            max_batch_size: Maximum batch size for inference
        """
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size

        # Create logger and load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_bytes = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if not self.engine:
            raise RuntimeError(f"Failed to load engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.bindings = []
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT

            if is_input:
                self.input_name = tensor_name
                self.input_dtype = dtype
                self.input_shape = shape
                print(f"  Input: {tensor_name}, shape: {shape}, dtype: {dtype}")
            else:
                self.output_name = tensor_name
                self.output_dtype = dtype
                self.output_shape = shape
                print(f"  Output: {tensor_name}, shape: {shape}, dtype: {dtype}")

        print("✓ Engine loaded successfully")

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Pre-allocate buffers for common batch size
        self._buffers = {}
        self._preallocate_buffer(1)  # Preallocate for batch size 1

    def _preallocate_buffer(self, batch_size):
        """Preallocate device buffers for a specific batch size."""
        if batch_size in self._buffers:
            return

        # Calculate shapes
        input_shape = (batch_size,) + tuple(self.input_shape[1:])
        self.context.set_input_shape(self.input_name, input_shape)
        output_shape = self.context.get_tensor_shape(self.output_name)

        # Calculate sizes
        input_size = int(np.prod(input_shape)) * np.dtype(self.input_dtype).itemsize
        output_size = int(np.prod(output_shape)) * np.dtype(self.output_dtype).itemsize

        # Allocate device memory
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        self._buffers[batch_size] = {
            'input_shape': input_shape,
            'output_shape': output_shape,
            'd_input': d_input,
            'd_output': d_output,
            'input_size': input_size,
            'output_size': output_size
        }

    def infer(self, input_data):
        """
        Run inference on input data.

        Args:
            input_data: numpy array or torch tensor with shape [B, L, N, 3]

        Returns:
            numpy array with shape [B, num_joints*3]
        """
        # Convert to numpy if needed
        if isinstance(input_data, torch.Tensor):
            input_np = input_data.cpu().numpy().astype(self.input_dtype)
        else:
            input_np = input_data.astype(self.input_dtype)

        batch_size = input_np.shape[0]

        # Allocate buffer if needed
        if batch_size not in self._buffers:
            self._preallocate_buffer(batch_size)

        # Get preallocated buffers
        buf = self._buffers[batch_size]

        # Update context with batch size
        self.context.set_input_shape(self.input_name, buf['input_shape'])

        # Prepare input
        h_input = np.ascontiguousarray(input_np)
        h_output = np.empty(buf['output_shape'], dtype=self.output_dtype)

        # Copy input to device
        cuda.memcpy_htod_async(buf['d_input'], h_input, self.stream)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(buf['d_input']))
        self.context.set_tensor_address(self.output_name, int(buf['d_output']))

        # Execute inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT execution failed")

        # Copy output from device
        cuda.memcpy_dtoh_async(h_output, buf['d_output'], self.stream)

        # Synchronize
        self.stream.synchronize()

        return h_output

    def infer_torch(self, input_tensor):
        """
        Run inference and return torch tensor.

        Args:
            input_tensor: torch tensor with shape [B, L, N, 3]

        Returns:
            torch tensor with shape [B, num_joints*3]
        """
        device = input_tensor.device
        output_np = self.infer(input_tensor)
        return torch.from_numpy(output_np).to(device)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        if hasattr(self, 'runtime'):
            del self.runtime


def test_inference():
    """Test the TensorRT inference."""
    import time

    # Create model
    engine_path = "/home/oliver/Documents/SPiKE/experiments/Custom/pretrained-full/log/best_model.engine"
    model = TensorRTModel(engine_path)

    # Create dummy input [batch, frames, points, xyz]
    batch_size = 1
    frames = 3
    points = 4096
    input_data = np.random.randn(batch_size, frames, points, 3).astype(np.float32)

    print(f"\nInput shape: {input_data.shape}")

    # Run inference
    print("\nRunning inference...")
    start = time.time()
    output = model.infer(input_data)
    end = time.time()

    print(f"✓ Inference completed in {(end-start)*1000:.2f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")

    # Benchmark
    print("\nBenchmarking (100 iterations)...")
    times = []
    for _ in range(10):  # Warmup
        _ = model.infer(input_data)

    for _ in range(100):
        start = time.time()
        _ = model.infer(input_data)
        times.append((time.time() - start) * 1000)

    print(f"Average: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
    print(f"FPS: {1000/np.mean(times):.2f}")


if __name__ == '__main__':
    test_inference()
