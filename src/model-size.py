!pip install pympler

from pympler import asizeof

# Assuming 'model' is your trained model
memory_usage = asizeof.asizeof(model)
print(f"Estimated model memory usage: {memory_usage / (1024 * 1024)} MB")  # Convert bytes to megabytes

# Save the model
model.save('gender_blog')

# Check the size of the saved model file
# import os
file_size = os.path.getsize('gender_blog')
print(f"Model file size: {file_size / (1024 * 1024)} MB")  # Convert bytes to megabytes