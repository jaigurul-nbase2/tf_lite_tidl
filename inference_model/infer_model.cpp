#include <iostream>
#include <memory>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <model.h>
#include <interpreter.h>
#include <kernels/register.h>
#include "mfcc_data.h"

typedef void (*ErrorHandler)(const char*);
typedef TfLiteDelegate* (*Create_delegate)(char**,
                                           char**,
                                           size_t,
                                           void (*report_error)(const char *));

int main() {
    // Path to the model
    const char* model_path = "model/model.tflite";
    
    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path);
    
    if (!model) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return -1;
    }
    
    std::cout << "Model loaded successfully from: " << model_path << std::endl;
    
    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    
    if (!interpreter) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return -1;
    }
    
    std::cout << "Interpreter created successfully" << std::endl;
    
    // Allocate tensor buffers FIRST (before delegate)
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }
    
    std::cout << "Tensors allocated successfully" << std::endl;
    
    // Load and configure TIDL delegate AFTER allocation
    bool enableTidl = true;  // Set to false to disable delegate
    if (enableTidl) {
        void* lib = dlopen("/usr/lib/libtidl_tfl_delegate.so", RTLD_NOW | RTLD_LOCAL);
        
        if (lib == NULL) {
            std::cerr << "Warning: Could not load TIDL delegate library" << std::endl;
            std::cerr << "Error: " << dlerror() << std::endl;
            std::cerr << "Continuing without delegate..." << std::endl;
        } else {
            std::cout << "TIDL delegate library loaded successfully" << std::endl;
            
            Create_delegate createPlugin = 
                (Create_delegate)dlsym(lib, "tflite_plugin_create_delegate");
            
            if (createPlugin == NULL) {
                std::cerr << "Warning: Could not find delegate creation function" << std::endl;
                std::cerr << "Error: " << dlerror() << std::endl;
                std::cerr << "Continuing without delegate..." << std::endl;
            } else {
                // Configure delegate options
                std::vector<const char*> keys;
                std::vector<const char*> values;
                
                // Artifacts folder - MUST exist and contain compiled artifacts
                keys.push_back("artifacts_folder");
                values.push_back("./classification/artifacts");  // Use relative or absolute path
                
                // Number of cores to use
                keys.push_back("num_tidl_subgraphs");
                values.push_back("1");
                
                // Debug level
                keys.push_back("debug_level");
                values.push_back("2");
                
                // Allow mixed precision
                keys.push_back("allow_mixed_precision");
                values.push_back("1");
                
                std::cout << "Creating TIDL delegate with artifacts from: " << values[0] << std::endl;
                
                TfLiteDelegate* delegate = createPlugin(
                    (char**)keys.data(), 
                    (char**)values.data(), 
                    keys.size(), 
                    NULL);
                
                if (delegate != NULL) {
                    std::cout << "Delegate created, applying to graph..." << std::endl;
                    
                    TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate);
                    if (status == kTfLiteOk) {
                        std::cout << "✓ TIDL delegate applied successfully to graph" << std::endl;
                        
                        // Re-allocate tensors after delegate modification
                        if (interpreter->AllocateTensors() != kTfLiteOk) {
                            std::cerr << "Failed to re-allocate tensors after delegate" << std::endl;
                            return -1;
                        }
                        std::cout << "✓ Tensors re-allocated after delegate" << std::endl;
                    } else {
                        std::cerr << "✗ Failed to apply delegate to graph (status: " << status << ")" << std::endl;
                        std::cerr << "  Check if artifacts folder exists and contains valid files" << std::endl;
                    }
                } else {
                    std::cerr << "✗ Failed to create delegate (returned NULL)" << std::endl;
                }
            }
        }
    }

    
    // // Allocate tensor buffers
    // if (interpreter->AllocateTensors() != kTfLiteOk) {
    //     std::cerr << "Failed to allocate tensors" << std::endl;
    //     return -1;
    // }
    
    // std::cout << "Tensors allocated successfully" << std::endl;
    
    // Print model information
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Number of inputs: " << interpreter->inputs().size() << std::endl;
    std::cout << "Number of outputs: " << interpreter->outputs().size() << std::endl;

    // input data to model
    std::vector<float> input_data(mfcc_data, mfcc_data + 940);

    // ADD THIS: Verify input data is not all zeros
    std::cout << "\n=== Input Data Verification ===" << std::endl;
    std::cout << "Input data size: " << input_data.size() << std::endl;
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << input_data[i] << " ";
    }
    std::cout << "\nLast 10 values: ";
    for (int i = input_data.size() - 10; i < input_data.size(); i++) {
        std::cout << input_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Print input tensor info
    for (size_t i = 0; i < interpreter->inputs().size(); i++) {
        const TfLiteTensor* input_tensor = interpreter->input_tensor(i);
        std::cout << "\nInput[" << i << "]:" << std::endl;
        std::cout << "  Name: " << input_tensor->name << std::endl;
        std::cout << "  Type: " << TfLiteTypeGetName(input_tensor->type) << std::endl;
        std::cout << "  Dimensions: " << input_tensor->dims->size << std::endl;
        std::cout << "  Shape: [";
        for (int j = 0; j < input_tensor->dims->size; j++) {
            std::cout << input_tensor->dims->data[j];
            if (j < input_tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Size (bytes): " << input_tensor->bytes << std::endl;
    }
    
    // Print output tensor info
    for (size_t i = 0; i < interpreter->outputs().size(); i++) {
        const TfLiteTensor* output_tensor = interpreter->output_tensor(i);
        std::cout << "\nOutput[" << i << "]:" << std::endl;
        std::cout << "  Name: " << output_tensor->name << std::endl;
        std::cout << "  Type: " << TfLiteTypeGetName(output_tensor->type) << std::endl;
        std::cout << "  Dimensions: " << output_tensor->dims->size << std::endl;
        std::cout << "  Shape: [";
        for (int j = 0; j < output_tensor->dims->size; j++) {
            std::cout << output_tensor->dims->data[j];
            if (j < output_tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Size (bytes): " << output_tensor->bytes << std::endl;
    }
    
    std::cout << "\n=== Model loaded and ready ===" << std::endl;
    
    // Copy input data to input tensor
    float* input_tensor_data = interpreter->typed_input_tensor<float>(0);
    if (input_tensor_data == nullptr) {
        std::cerr << "Failed to get input tensor data pointer" << std::endl;
        return -1;
    }
    
    // Calculate total input size
    const TfLiteTensor* input_tensor = interpreter->input_tensor(0);
    int total_input_size = 1;
    for (int i = 0; i < input_tensor->dims->size; i++) {
        total_input_size *= input_tensor->dims->data[i];
    }
    if (total_input_size != input_data.size()) {
        std::cerr << "ERROR: Input size mismatch! Expected " << total_input_size 
                  << " but got " << input_data.size() << std::endl;
        return -1;
    }
    
    std::memcpy(input_tensor_data,
                input_data.data(),
                input_data.size() * sizeof(float));
    // ADD THIS: Verify data was copied correctly
    std::cout << "First 10 values in tensor after copy: ";
    for (int i = 0; i < 10; i++) {
        std::cout << input_tensor_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== Running inference ===" << std::endl;
    // ADD THIS: Single inference for debugging
    std::cout << "Running single test inference..." << std::endl;
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Failed to invoke interpreter" << std::endl;
        return -1;
    }
    
    // Check output immediately
    float* test_output = interpreter->typed_output_tensor<float>(0);
    std::cout << "Output after first inference: ";
    for (int i = 0; i < 10; i++) {
        std::cout << test_output[i] << " ";
    }
    std::cout << std::endl;
    
    
    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run inference multiple times to get accurate timing
    const int num_iterations = 10;
    for (int iter = 0; iter < num_iterations; iter++) {
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke interpreter" << std::endl;
            return -1;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Total time for " << num_iterations << " iterations: " 
              << duration.count() << " ms" << std::endl;
    std::cout << "Average inference time: " 
              << duration.count() / (float)num_iterations << " ms" << std::endl;
    std::cout << "FPS: " 
              << (num_iterations * 1000.0f) / duration.count() << std::endl;
    
    // Get output results
    std::cout << "\n=== Output Results ===" << std::endl;
    for (size_t i = 0; i < interpreter->outputs().size(); i++) {
        const TfLiteTensor* output_tensor = interpreter->output_tensor(i);
        std::cout << "\nOutput[" << i << "] - " << output_tensor->name << ":" << std::endl;
        
        // Calculate total elements
        int total_elements = 1;
        for (int j = 0; j < output_tensor->dims->size; j++) {
            total_elements *= output_tensor->dims->data[j];
        }
        
        // Print output values based on type
        if (output_tensor->type == kTfLiteFloat32) {
            float* output_data = interpreter->typed_output_tensor<float>(i);
            std::cout << "  Values: [";
            int max_print = std::min(total_elements, 20); // Print first 20 values max
            for (int j = 0; j < max_print; j++) {
                std::cout << output_data[j];
                if (j < max_print - 1) std::cout << ", ";
            }
            if (total_elements > 20) {
                std::cout << ", ... (" << total_elements << " total elements)";
            }
            std::cout << "]" << std::endl;
            
            // For classification, find the class with max probability
            if (total_elements > 1) {
                int max_idx = 0;
                float max_val = output_data[0];
                for (int j = 1; j < total_elements; j++) {
                    if (output_data[j] > max_val) {
                        max_val = output_data[j];
                        max_idx = j;
                    }
                }
                std::cout << "  Predicted class: " << max_idx << " (score: " << max_val << ")" << std::endl;
            }
        } else if (output_tensor->type == kTfLiteUInt8) {
            uint8_t* output_data = interpreter->typed_output_tensor<uint8_t>(i);
            std::cout << "  Values: [";
            int max_print = std::min(total_elements, 20);
            for (int j = 0; j < max_print; j++) {
                std::cout << (int)output_data[j];
                if (j < max_print - 1) std::cout << ", ";
            }
            if (total_elements > 20) {
                std::cout << ", ... (" << total_elements << " total elements)";
            }
            std::cout << "]" << std::endl;
        } else if (output_tensor->type == kTfLiteInt8) {
            int8_t* output_data = interpreter->typed_output_tensor<int8_t>(i);
            std::cout << "  Values: [";
            int max_print = std::min(total_elements, 20);
            for (int j = 0; j < max_print; j++) {
                std::cout << (int)output_data[j];
                if (j < max_print - 1) std::cout << ", ";
            }
            if (total_elements > 20) {
                std::cout << ", ... (" << total_elements << " total elements)";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    std::cout << "\n=== Inference complete ===" << std::endl;
    
    return 0;
}