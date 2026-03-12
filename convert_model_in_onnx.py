import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

def convert_onnx_to_h5(onnx_path, h5_path):
    print(f"Loading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(onnx_path)
    
    print("Preparing TensorFlow representation...")
    tf_rep = prepare(onnx_model)
    
    # Export to SavedModel first
    temp_pb_path = "temp_tf_model"
    print(f"Exporting to temporary SavedModel at {temp_pb_path}...")
    tf_rep.export_graph(temp_pb_path)
    
    print("Loading SavedModel into Keras...")
    model = tf.keras.models.load_model(temp_pb_path)
    
    print(f"Saving Keras model to {h5_path}...")
    model.save(h5_path)
    
    # Cleanup
    import shutil
    if os.path.exists(temp_pb_path):
        shutil.rmtree(temp_pb_path)
    
    print("Conversion complete!")

if __name__ == "__main__":
    onnx_file = "facial_expression_recognition_mobilefacenet.onnx"
    h5_file = "MUL_KSIZE_MobileNet_v2_bext.hdf5"
    
    if os.path.exists(onnx_file):
        try:
            convert_onnx_to_h5(onnx_file, h5_file)
        except ImportError:
            print("Error: Please install 'onnx', 'onnx-tf', and 'tensorflow'.")
            print("Run: pip install onnx onnx-tf tensorflow")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print(f"Error: {onnx_file} not found.")
