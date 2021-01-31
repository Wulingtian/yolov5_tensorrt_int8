#ifndef __MY_INTERFACE_H_
#define __MY_INTERFACE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "my_common.h"

   result_t my_init_tensors(tensor_params_array_t *input_tensors_params, tensor_params_array_t *output_tensors_params,
                         tensor_array_t **input_tensors, tensor_array_t **output_tensors);

   result_t my_deinit_tensors(tensor_array_t *input_tensors, tensor_array_t *output_tensors);


   result_t my_load_model(model_params_t *load_model_param,
                       tensor_array_t *input_tensors,
                       tensor_array_t *output_tensors,
                       model_handle_t *load_model_handle);


   result_t my_release_model(model_handle_t *load_model_handle);


   result_t  my_inference_tensors(model_handle_t *load_model_handle);

#ifdef __cplusplus
}
#endif

#endif