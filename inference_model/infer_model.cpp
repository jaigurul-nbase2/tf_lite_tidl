#include <iostream>
#include <memory>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <sys/stat.h>
#include <model.h>
#include <interpreter.h>
#include <kernels/register.h>
#include "mfcc_data.h"

typedef void (*ErrorHandler)(const char*);
typedef TfLiteDelegate* (*Create_delegate)(char**,
                                           char**,
                                           size_t,
                                           void (*report_error)(const char *));

// TIDL error reporter callback
void tidl_error_reporter(const char* msg) {
    std::cerr << "TIDL: " << msg << std::endl;
}

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
    
    // Store original node count BEFORE delegate
    int original_node_count = 0;
    
    // Allocate tensor buffers FIRST (before delegate)
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return -1;
    }
    
    std::cout << "Tensors allocated successfully" << std::endl;
    
    // Get node count before delegation
    original_node_count = interpreter->execution_plan().size();
    std::cout << "Original graph has " << original_node_count << " operations" << std::endl;
    
    // Load and configure TIDL delegate AFTER allocation
    bool enableTidl = false;  // Set to false to disable delegate
    bool tidl_delegate_applied = false;
    int delegated_node_count = 0;
    
    if (enableTidl) {
        // First, verify artifacts folder exists
        std::cout << "\n=== Verifying TIDL Artifacts ===" << std::endl;
        const char* artifacts_path = "./classification/artifacts";
        struct stat info;
        
        if (stat(artifacts_path, &info) != 0) {
            std::cerr << "ERROR: Artifacts folder does not exist: " << artifacts_path << std::endl;
            std::cerr << "Please run TIDL compilation first to generate artifacts" << std::endl;
            enableTidl = false;
        } else if (!(info.st_mode & S_IFDIR)) {
            std::cerr << "ERROR: " << artifacts_path << " is not a directory!" << std::endl;
            enableTidl = false;
        } else {
            std::cout << "✓ Artifacts folder found: " << artifacts_path << std::endl;
        }
    }
    
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
                values.push_back("./classification/artifacts");
                
                // Number of cores to use
                keys.push_back("num_tidl_subgraphs");
                values.push_back("1");
                
                // Debug level - 0 for production, 1 for minimal logging, 3 for verbose
                keys.push_back("debug_level");
                values.push_back("0");
                
                // Allow mixed precision
                keys.push_back("allow_mixed_precision");
                values.push_back("1");
                
                std::cout << "Creating TIDL delegate with artifacts from: " << values[0] << std::endl;
                std::cout << "Debug level: " << values[2] << std::endl;
                
                // Create delegate with error reporter
                TfLiteDelegate* delegate = createPlugin(
                    (char**)keys.data(), 
                    (char**)values.data(), 
                    keys.size(), 
                    tidl_error_reporter);
                
                if (delegate != NULL) {
                    std::cout << "Delegate created, applying to graph..." << std::endl;
                    
                    TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate);
                    if (status == kTfLiteOk) {
                        std::cout << "✓ TIDL delegate applied successfully to graph" << std::endl;
                        tidl_delegate_applied = true;
                        
                        // Re-allocate tensors after delegate modification
                        if (interpreter->AllocateTensors() != kTfLiteOk) {
                            std::cerr << "Failed to re-allocate tensors after delegate" << std::endl;
                            return -1;
                        }
                        std::cout << "✓ Tensors re-allocated after delegate" << std::endl;
                        
                        // FIXED: Better delegation verification that works across TF Lite versions
                        std::cout << "\n=== Delegation Verification ===" << std::endl;
                        
                        int final_node_count = interpreter->execution_plan().size();
                        int delegate_nodes = 0;
                        int cpu_nodes = 0;
                        
                        std::cout << "Analyzing execution plan..." << std::endl;
                        std::cout << "Graph before delegation: " << original_node_count << " nodes" << std::endl;
                        std::cout << "Graph after delegation: " << final_node_count << " nodes" << std::endl;
                        
                        for (size_t i = 0; i < interpreter->execution_plan().size(); i++) {
                            int node_index = interpreter->execution_plan()[i];
                            auto* node_and_reg = interpreter->node_and_registration(node_index);
                            const TfLiteRegistration& reg = node_and_reg->second;
                            
                            // Multiple ways to detect delegate nodes for compatibility
                            bool is_delegate = false;
                            
                            // Method 1: Check for custom_name containing "Delegate" or "TIDL"
                            if (reg.custom_name) {
                                std::string name_str(reg.custom_name);
                                if (name_str.find("Delegate") != std::string::npos ||
                                    name_str.find("TIDL") != std::string::npos ||
                                    name_str.find("TfLiteDelegate") != std::string::npos) {
                                    is_delegate = true;
                                }
                            }
                            
                            // Method 2: Check builtin_code (if available in this version)
                            // BuiltinOperator_DELEGATE = 0 in some versions
                            if (!is_delegate && reg.builtin_code == 0) {
                                // builtin_code 0 could be DELEGATE, but need to verify it's not something else
                                // Check if it has no standard name
                                if (reg.custom_name || (reg.builtin_code < tflite::BuiltinOperator_MAX && 
                                    strcmp(tflite::EnumNamesBuiltinOperator()[reg.builtin_code], "DELEGATE") == 0)) {
                                    is_delegate = true;
                                }
                            }
                            
                            if (is_delegate) {
                                delegate_nodes++;
                                std::cout << "  Node[" << i << "]: TIDL DELEGATE NODE";
                                if (reg.custom_name) {
                                    std::cout << " (" << reg.custom_name << ")";
                                }
                                std::cout << std::endl;
                            } else {
                                cpu_nodes++;
                                const char* op_name = "UNKNOWN";
                                if (reg.custom_name) {
                                    op_name = reg.custom_name;
                                } else if (reg.builtin_code < tflite::BuiltinOperator_MAX) {
                                    op_name = tflite::EnumNamesBuiltinOperator()[reg.builtin_code];
                                }
                                std::cout << "  Node[" << i << "]: " << op_name << " (CPU)" << std::endl;
                            }
                        }
                        
                        std::cout << "\n--- Delegation Summary ---" << std::endl;
                        std::cout << "Original operations: " << original_node_count << std::endl;
                        std::cout << "Current execution plan: " << final_node_count << " nodes" << std::endl;
                        std::cout << "  Delegate nodes: " << delegate_nodes << std::endl;
                        std::cout << "  CPU-only nodes: " << cpu_nodes << std::endl;
                        
                        // Calculate estimated delegation
                        // When nodes are delegated, they're replaced by delegate node(s)
                        // delegated_ops ≈ original_count - final_count + delegate_nodes
                        // Or more simply: original_count - cpu_nodes (if we assume delegate replaced ops)
                        
                        if (delegate_nodes > 0) {
                            // If we have delegate nodes and the graph shrank, operations were delegated
                            delegated_node_count = original_node_count - cpu_nodes;
                            float delegation_percent = (100.0f * delegated_node_count) / original_node_count;
                            
                            std::cout << "\nEstimated delegated operations: " << delegated_node_count 
                                     << " (~" << delegation_percent << "%)" << std::endl;
                            
                            if (cpu_nodes == 0) {
                                std::cout << "\n✓ Perfect! All operations delegated to TIDL accelerator." << std::endl;
                            } else {
                                std::cout << "\n✓ Partial delegation: Most ops on TIDL, " << cpu_nodes << " on CPU." << std::endl;
                                std::cout << "  This is normal for operations not supported by TIDL." << std::endl;
                            }
                        } else {
                            std::cerr << "\n⚠️  WARNING: No delegate nodes detected in execution plan!" << std::endl;
                            std::cerr << "    However, TIDL reports delegation succeeded." << std::endl;
                            std::cerr << "    Delegation may still be working (check performance)." << std::endl;
                        }
                        std::cout << "-------------------------\n" << std::endl;
                        
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
    
    // Print model information
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Number of inputs: " << interpreter->inputs().size() << std::endl;
    std::cout << "Number of outputs: " << interpreter->outputs().size() << std::endl;
    std::cout << "TIDL Delegate: " << (tidl_delegate_applied ? "ENABLED ✓" : "DISABLED") << std::endl;
    if (tidl_delegate_applied && delegated_node_count > 0) {
        std::cout << "Accelerated operations: ~" << delegated_node_count << "/" << original_node_count << std::endl;
    }

    // input data to model
    std::vector<float> input_data(mfcc_data, mfcc_data + 940);

    // Verify input data
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
    
    // Verify data was copied correctly
    std::cout << "First 10 values in tensor after copy: ";
    for (int i = 0; i < 10; i++) {
        std::cout << input_tensor_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n=== Running inference ===" << std::endl;
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
    const int num_iterations = 100;
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
    std::cout << "Mode: " << (tidl_delegate_applied ? "TIDL Accelerated" : "CPU Only") << std::endl;
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