#include "Inference.h"

void Model_Inference::load_model(string model_path)
{
	graph::SetDefaultDevice("/gpu:0", &graphdef);
	session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
	session_options.config.mutable_gpu_options()->set_allow_growth(true);
	
	cout << "model path is !" << model_path << endl;
	auto status = LoadSavedModel(session_options, run_options, model_path, { "serve" }, &bundle);
	if (!status.ok())
	{
		cout << "ERROR: Loading model to graph failed successfully!\n" << flush;
		cout << status.ToString() << flush;
	}
	else
	{
		cout << "INFO: Loading model to graph successfully !\n" << flush;
	}
	session->Create(graphdef);
}

void Model_Inference::convertCVMatToTensor(Mat srcimage,int width, int height)
{
	input_tensor = Tensor(DT_FLOAT, TensorShape({ 1, width, height, 1 }));
	srcimage.convertTo(srcimage, CV_32FC(srcimage.channels()));
	float *tensor_data_ptr = input_tensor.flat<float>().data();
	Mat output_mat(width, height, CV_32FC(srcimage.channels()), tensor_data_ptr);
	srcimage.convertTo(output_mat, CV_32FC(srcimage.channels()));
}


int Model_Inference::predict(Mat scrimage, int height, int width)
{
	int i = 0;
	vector<Tensor> out_tensors;

	if (scrimage.empty())
	{
		cout << "can't open the image!!!!!!! " << "\n" <<flush;
		//return -1;
	}
	else
	{
		cout << " open the image successfully!!!!!!! "<< "\n"<<flush;
	}

	convertCVMatToTensor(scrimage, height, width); // input image -> input tensor
	auto status = session->Run({ {tensor_input,input_tensor} }, { tensor_output }, {}, &out_tensors);
	if (!status.ok())
	{
		cout << "error: run failed...\n" << flush;
		cout << status.ToString() + "\n" << flush;
		//return -1;
	}
	else
	{
		cout << status.ToString() << flush;
	}
	cout << "Output tensor size" << out_tensors.size() << "\n" << flush;

	TTypes<float>::Flat pp = out_tensors[0].flat<float>();
	vector<float> pp_res;
	for (i = 0; i < pp.size(); i++)
	{
		pp_res.push_back(pp(i));
	}
	conf = *max_element(pp_res.begin(), pp_res.end());
	maxidx = max_element(pp_res.begin(), pp_res.end()) - pp_res.begin();
	return 0;
}


INFERENCE_API void Initial_infer()
{
	CNN.load_model("../Debug/save_model/my_model/");
}


INFERENCE_API char* model_pred(unsigned char* imgptr, double params[3])
{
	try
	{
		Mat img_dst;
		int img_W = params[0];
		int img_H = params[1];
		size_t stride = params[2];
		int re_width = 54;      // image resize width
		int re_height = 54;    // image resize height
		CNN.input_pic = Mat(img_H, img_W, CV_8UC1, imgptr, stride);
		resize(CNN.input_pic, img_dst, cv::Size(re_height, re_width), 0, 0, INTER_LINEAR);
		CNN.predict(img_dst, re_width, re_height);
		//CNN.predict(CNN.input_pic, re_width, re_height);
		return &CNN.class_list[maxidx][0];
	}
	catch (const exception& ex)
	{
		return &std::string(ex.what())[0];
	}
}