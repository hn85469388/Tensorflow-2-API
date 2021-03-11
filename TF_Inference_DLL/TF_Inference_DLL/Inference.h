#pragma once

#define COMPILER_MSVC
#define NOMINMAX

//#ifndef INFERENCEDLL_EXPORTS
#define INFERENCE_API __declspec( dllexport ) 
//#else
//#define INFERENCE_API __declspec( dllimport)
//#endif 




#include<iostream>
#include<fstream>
//--------------- TensorFlow LIB --------------------
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/tensor.h"
//------------------Other Lib----------------------------
//#include "nlohmann/json.hpp"
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>




using namespace cv;
using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;
using tensorflow::SavedModelBundle;
using tensorflow::SessionOptions;
using tensorflow::RunOptions;
using tensorflow::Scope;
using tensorflow::ClientSession;
using tensorflow::MetaGraphDef;



class Model_Inference
{
public:
	Mat input_pic;
	string model_path;
	vector <string>class_list = { "11", "22", "33", "44" };
	void load_model(string model_path);
	int predict(Mat input_pic, int height, int width);
private:
	SavedModelBundle bundle;
	SessionOptions session_options;
	RunOptions run_options;
	Tensor input_tensor;                          //Mat convert Tensor format
	GraphDef graphdef;
	unique_ptr<Session>& session = bundle.session;
	
	string tensor_input = "serving_default_conv2d_input";
	string tensor_output = "StatefulPartitionedCall:0";
	void convertCVMatToTensor(Mat srcImage, int height, int width);

}CNN;
static float conf;
static int maxidx;

extern "C" INFERENCE_API void Initial_infer();
extern "C" INFERENCE_API char* model_pred(unsigned char* imgptr, double params[3]);