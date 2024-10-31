import os
import pathlib
import time
import numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

# Number of iterations
num_iterations = 100

# Get the current script directory
script_dir = pathlib.Path(__file__).parent.absolute()

# model directory
directory='div_4'

# List of model filenames
model_files = [f for f in os.listdir(os.path.join(script_dir, directory)) if f.endswith('.tflite')]

# Loop through each model
for model_file in model_files:
    # Load the model file path
    model_path = os.path.join(script_dir, directory + '/' + model_file)

    # Initialize the TF interpreter
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()

    # Get input tensor details
    input_details = interpreter.get_input_details()

    # Extract the shape and dtype from the input tensor
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Run inference 10 times and calculate the average inference time
    total_time = 0
    interpreter.invoke()

    for _ in range(num_iterations):
        # Create a zero-filled NumPy array with the correct shape and dtype
        input_data = np.zeros(input_shape, dtype=input_dtype)
        
        # Set the input tensor
        common.set_input(interpreter, input_data)
        
        # Measure inference time
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        
        # Accumulate total time
        total_time += inference_time

    # Calculate average inference time
    average_time = total_time / num_iterations

    # Print the result
    print(f'{model_file} {average_time * 1000:.1f} ms')
