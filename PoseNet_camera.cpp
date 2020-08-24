#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <map>
#include <fstream>
#include <sys/time.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <signal.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
/*****************************************
* Includes
******************************************/
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <errno.h>
#include <jpeglib.h>
#include <termios.h>
#include <math.h>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

#include "define.h"

using namespace cv; 
using namespace std;

/*****************************************
* Macros definition
******************************************/
#define RESNET_str "resnet50"
#define MOBILENET_str "mobilenet"
#define NUM_KEYPOINTS 17
#define NUM_CHAIN 16
#define LOCAL_MAXIMUM_RADIUS 1
#define MAX_POSE_DETECTIONS 10
#define MIN_POSE_SCORE 0.25
#define MODEL_01 1   // this model has input shape (1, 3, 129, 129)
#define MODEL_02 2   // this model has input shape (1, 3, 257, 257)
#define MODEL_03 3   // this model has input shape (1, 3, 513, 513)
#define DISABLE 0
#define ENABLE 1

/*****************************************
* Global Variables
******************************************/
int model=RESNET50;
std::string model_name = MOBILENET_str;
std::string stride_name;
char image_size[10];
float score_threshold = 0.5;
std::map<int,std::string> label_file_map;
std::map<int,std::string> label_chain_map;
int input_image_size = 257;
int tget_wid;
int tget_hei;
int image_size_x;
int image_size_y;
int stride = 16; // default stride is 16
int cam_index = 8; //device name of  camera is /dev/video<cam_index>
int arr_size;
int model_index = 2;
int headmap_id,offset_id,bwd_id,fwd_id;
const char* mat_out = "mat_out.jpg";
std::string part_names_file("part_names.txt");
std::string chain_names_file("chain_names.txt");
// ONNX Runtime variables
OrtEnv* env;
OrtSession* session;
OrtSessionOptions* session_options;
size_t num_input_nodes;
size_t num_output_nodes;
OrtStatus* status;
float* out_data[4];// = NULL;
int measure_time = 0;
int print_poses_score = 0; //skip print pose result in case run real time with camera to increase performance, use it for to debug only
int img_sizex, img_sizey, img_channels;
cv::Mat camera_frame;

std::vector<const char*> input_node_names(1);
std::vector<const char*> output_node_names(4);
std::vector<int64_t> input_node_dims_input;
std::vector<int64_t> input_node_dims_shape;
std::vector<int64_t> output_node_dims;
std::vector<OrtValue* > input_tensor(input_node_names.size());

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

struct S_Pixel
{
    unsigned char RGBA[3];
};

void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

/*****************************************
* Function Name : sigmoid
* Description   : helper function for Posenet Post Processing
* Arguments :
* Return value  :
******************************************/
float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
}

/*****************************************
* Function Name : timedifference_msec
* Description   :
* Arguments :
* Return value  :
******************************************/
static double timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

/*****************************************
* Function Name :  loadLabelFile
* Description       : Load txt file
* Arguments         :
* Return value  :
******************************************/
int loadLabelFile(std::string label_file_name,std::string label_chain_name)
{
    int counter_part = 0;
    int counter_chain = 0;
    std::ifstream infile(label_file_name);
    std::ifstream inchain(label_chain_name);

    if (!infile.is_open() || !inchain.is_open())
    {
        perror("error while opening file");
        return -1;
    }

    std::string line_file, line_chain;
    while(std::getline(infile,line_file))
    {
        label_file_map[counter_part++] = line_file;
    }
    while(std::getline(inchain,line_chain))
    {
        label_chain_map[counter_chain++] = line_chain;
    }

    if (infile.bad() || inchain.bad())
    {
        perror("error while reading file");
        return -1;
    }
    return 0;
}


/*****************************************
* Function Name : help function
* Description   :
* Arguments :
* Return value  :
******************************************/
int help()
{   
    int ret = 0;
    printf("\n");
    printf("To run posenet app, Please add below input arguments: \n");
    printf("    Model for posenet: -model <model name> \n");
    printf("        <model name> can be mobilenet or resnet50, default is mobilenet \n");
    printf("    Model index for posenet: -model_index <model index> \n");
    printf("        <model index>  can be 1 or 2 or 3, default is 2 \n");
    printf("    Camera index for posenet: -cam_index <cam_index> \n");
    printf("        <cam_index>  is index of device name camera: /dev/video<cam_index>  \n");
    printf("    Input image size for posenet: -measure_time <measure_time> \n");
    printf("        <measure_time>  to enable/disable the measure time function. 0 for disable, 1 for enable \n");
    printf("    Input image size for posenet: -print_poses_score <print_poses_score> \n");
    printf("        <print_poses_score>  to enable/disable the print pose score function. 0 for disable, 1 for enable \n");
    printf("Example command: ./poseNet_camera -model mobilenet -model_index 2 -cam_index 8 -measure_time 0 -print_poses_score 0\n");
    printf("\n");

    return ret;
}
/*****************************************
* Function Name : print information function
* Description   :
* Arguments :
* Return value  :
******************************************/
int print_info()
{   
    int ret = 0;
    printf("\n");
    printf("There are 3 model of mobilenet: \n");
    printf("Model 1:\n");
    printf("    stride: 16 \n");
    printf("    input image size: 129x129 \n");
    printf("Model 2:\n");
    printf("    stride: 16 \n");
    printf("    input image size: 257x257 \n");
    printf("Model 3:\n");
    printf("    stride: 16 \n");
    printf("    input image size: 513x513 \n");
    printf("\n");

    return ret;
}

/*****************************************
* Function Name : parse_argument
* Description   :
* Arguments :
* Return value  :
******************************************/
int parse_argument(int argc, char* argv[])
{   
    int ret = 0;
	int index = 1;
    if(argc == 1)
    {
        printf("\n");
        printf("Lack of argument for posenet app. \n");
        printf("To get instruction, Run command: ./poseNet_camera -h\n");
        printf("To get model information, Run command: ./poseNet_camera -a\n");
        printf("\n");
        return 1;
    }
	for (index = 1; index < argc; index++) {
		if (!strcmp("-model", argv[index])) {
			model_name = argv[index+1];
		} else if (!strcmp("-model_index", argv[index])) {
			model_index = atoi(argv[index+1]);
		} else if (!strcmp("-cam_index", argv[index])) {
			cam_index = atoi(argv[index+1]);
		} else if (!strcmp("-measure_time", argv[index])) {
			measure_time = atoi(argv[index+1]);
		} else if (!strcmp("-print_poses_score", argv[index])) {
			print_poses_score = atoi(argv[index+1]);
		} else if (!strcmp("-h", argv[index])) {
			help();
            return 1;
		} else if (!strcmp("-a", argv[index])) {
			print_info();
            return 1;
        } else {
        }
    }
    return ret;
}

int prepare_environment()
{
    int ret = 0;
    
    // process some input argument
    if (!strcmp(model_name.c_str(), RESNET_str)) {
        model = RESNET50;
        printf("Currently, posenet app doesn't support Resnet architecture\n");
        return 1;
    }
    else if (!strcmp(model_name.c_str(), MOBILENET_str)) {
        model = MOBILENET;
        switch(model_index)
        {
            case MODEL_01:
                stride = 16;
                input_image_size = 129;
            break;
            case MODEL_02:
                stride = 16;
                input_image_size = 257;
            break;
            case MODEL_03:
                stride = 16;
                input_image_size = 513;
            break;
            default:
                printf("Currently, posenet app doesn't support model %d\n",model_index);
                return 1;
            break;
        }
    }
    else {
        printf("The architecture is not supported\n");
        ret = -1;
    }

    if((measure_time != DISABLE) && (measure_time != ENABLE))
    {
        printf("Wrong input of measure_time\n");
        return 1;
    }
    if((print_poses_score != DISABLE) && (print_poses_score != ENABLE))
    {
        printf("Wrong input of print_poses_score\n");
        return 1;
    }
    
    if(stride == 16) stride_name = "_stride16";
    else if(stride == 8) stride_name = "_stride8";
    else if(stride == 32) stride_name = "_stride32";

    sprintf(image_size,"%d",input_image_size);

    tget_hei = input_image_size;
    tget_wid = input_image_size;

    DIR* dir = opendir("output");
    if (!dir) ret =mkdir("output", 0777);
    
    if(loadLabelFile(part_names_file,chain_names_file) != 0)
    {
        fprintf(stderr,"Fail to open or process file %s, %s\n",part_names_file.c_str(),chain_names_file.c_str());
        return -1;
    }

    return ret;
}

// valid resolution
void valid_resolution(int width, int height, int *target_width, int *target_height) {
    *target_width = (width / stride) * stride +1;
    *target_height = (height / stride) * stride +1;
}
void prepare_ONNX_Runtime() {
    CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    g_ort->CreateSessionOptions(&session_options);
    g_ort->SetInterOpNumThreads(session_options, 2); //Multi-core

    //Config : model
    std::string onnx_model_name = model_name + stride_name + "_imagesize" + image_size + ".onnx";
    std::string onnx_model_path = "./models/" + onnx_model_name;

    //ONNX runtime load model
    CheckStatus(g_ort->CreateSession(env, onnx_model_path.c_str(), session_options, &session));
    printf("Start Loading Model %s\n", model_name.c_str());
}

int preprocess_input(VideoCapture cap) {
    int ret=0;

    cv::Mat img_ori;
    cap >> img_ori;

    int img_size_max;
    if(img_ori.rows > img_ori.cols) img_size_max = img_ori.rows;
    else img_size_max = img_ori.cols;

    cv::Mat _img(img_size_max,img_size_max, CV_8UC3, Scalar(128,128,128));
    img_ori.copyTo(_img(cv::Rect((img_size_max - img_ori.cols)/2,(img_size_max - img_ori.rows)/2,img_ori.cols, img_ori.rows)));
    camera_frame = _img.clone();

    // process image
    cv::Mat img;
    cv::resize(_img, img, cv::Size(tget_wid, tget_hei), 0, 0, CV_INTER_LINEAR);
    //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::imwrite(mat_out, img);
    stbi_uc * img_data = stbi_load(mat_out, &img_sizex, &img_sizey, &img_channels, STBI_default);

    const S_Pixel * imgPixels(reinterpret_cast<const S_Pixel *>(img_data));

    int input_tensor_size = img_sizex * img_sizey * img_channels;
    
    std::vector<float> input_tensor_values(input_tensor_size);

    arr_size = ((tget_wid - 1) / stride) + 1;
    if(model == MOBILENET)
    {
        int offs = 0;
        for (int c = 0; c < img_channels; c++){
            for (int y = 0; y < img_sizey; y++){
                for (int x = 0; x < img_sizex; x++, offs++){
                    const int val(imgPixels[y * img_sizex + x].RGBA[c]);
                    input_tensor_values[offs] = ((float)val)*2/255 - 1; // for mobilenet
                }
            }
        }
    }
    else if(model == RESNET50)
    {
        float image_net_mean[3] = {-123.15, -115.90, -103.06 };
        int offs = 0;
        for (int c = 0; c < img_channels; c++){
            for (int y = 0; y < img_sizey; y++){
                for (int x = 0; x < img_sizex; x++, offs++){
                    const int val(imgPixels[y * img_sizex + x].RGBA[c]);
                    input_tensor_values[offs] = (float)val + image_net_mean[c]; // for resnet
                }
            }
        }
    }

    // create input tensor object from data values
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // print model input layer (node names, types, shape etc.)    
    OrtAllocator* allocator;
    CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
    status = g_ort->SessionGetInputCount(session, &num_input_nodes);
    status = g_ort->SessionGetOutputCount(session, &num_output_nodes);

    // print input tensor type before setting value 
    for (size_t i = 0; i < num_input_nodes; i++){
        // print input node names
        char* input_name;
        status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
        //printf("Input %zu : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        OrtTypeInfo* typeinfo;
        status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        //printf("Input %zu : type=%d\n", i, type);
        size_t num_dims;

        if(i == 0)
        {
            num_dims = 4;
            //printf("Input %zu : num_dims=%zu\n", i, num_dims);
            input_node_dims_input.resize(num_dims);
            g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims_input.data(), num_dims);
            //check input dim
            
            //this is fixed model, so skip the update input value step
            //input_node_dims_input[0]=1;
            if(input_node_dims_input[1] == -1) input_node_dims_input[1]= tget_wid;
            if(input_node_dims_input[2] == -1) input_node_dims_input[2]= tget_hei;
            //input_node_dims_input[3]= img_channels;
            //check input dim
            //for (size_t j = 0; j < num_dims; j++) printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims_input[j]);
        }
        else  {
            //printf("incorrect input tensor dim count is %zu", i);
        }
        
        g_ort->ReleaseTypeInfo(typeinfo);
    }    
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        // print input node names
        char* output_name;
        CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
        CheckStatus(g_ort->SessionGetOutputName(session, i, allocator, &output_name));
        //printf("output %zu : name=%s\n", i, output_name);
        if(strstr(output_name, "heatmap") != NULL) headmap_id = i;
        else if(strstr(output_name, "offset") != NULL) offset_id = i;
        else if(strstr(output_name, "bwd") != NULL) bwd_id = i;
        else if(strstr(output_name, "fwd") != NULL) fwd_id = i;
        output_node_names[i] = output_name;
        // print input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo,&tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort->GetTensorElementType(tensor_info,&type));
        // print input shapes/dims
        size_t num_dims = 4;
        //printf("Output %zu : type=%d\n", i, type);
        //printf("Output %zu : num_dims=%zu\n", i, num_dims);
        output_node_dims.resize(num_dims);
        g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
        for (size_t j = 0; j < num_dims; j++) 
        {
            if(output_node_dims[j] == -1) output_node_dims[j] = arr_size;
            //printf("output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);
        }
        g_ort->ReleaseTypeInfo(typeinfo);
    }

    // set value for input tensor
    CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size*sizeof(float), input_node_dims_input.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[0]));
    int is_tensor;
    CheckStatus(g_ort->IsTensor(input_tensor[0],&is_tensor));
    assert(is_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    return ret;
}

int postprocess(VideoWriter video) 
{
    int ret = 0;
    int lmd = 2 * LOCAL_MAXIMUM_RADIUS + 1;
    float arr_heatmap[arr_size][arr_size][NUM_KEYPOINTS];
    float arr_heatmap_temp[NUM_KEYPOINTS][arr_size][arr_size];
    float arr_offset[arr_size][arr_size][NUM_KEYPOINTS*2];
    float arr_offset_temp[NUM_KEYPOINTS*2][arr_size][arr_size];
    float arr_fwd[arr_size][arr_size][NUM_CHAIN*2];
    float arr_fwd_temp[NUM_CHAIN*2][arr_size][arr_size];
    float arr_bwd[arr_size][arr_size][NUM_CHAIN*2];
    float arr_bwd_temp[NUM_CHAIN*2][arr_size][arr_size];
    float kp_scores[arr_size][arr_size];
    float max_vals[arr_size][arr_size];
    int max_loc[arr_size][arr_size];
    float parts[arr_size*arr_size*NUM_KEYPOINTS][4];
    float max_heat=0;
    int square_x1;
    int square_x2;
    int square_y1;
    int square_y2;
    int flag = 0;
    int part_num = 0;
    float root_score;
    int root_id;
    float root_coord[2];
    float root_image_coords[2];
    int pose_count = 0;
    float instance_keypoint_scores[NUM_KEYPOINTS];
    float instance_keypoint_coords[NUM_KEYPOINTS][2];
    float pose_scores[MAX_POSE_DETECTIONS];
    float pose_keypoint_scores[MAX_POSE_DETECTIONS][NUM_KEYPOINTS];
    float pose_keypoint_coords[MAX_POSE_DETECTIONS][NUM_KEYPOINTS][2];
    int squared_nms_radius;
    int nms_radius = 20;
    int num_parts = label_file_map.size();;
    int num_edges = label_chain_map.size()/2;
    int target_keypoint_id, source_keypoint_id;
    float source_keypoint_indices[2];
    float displaced_point[2];
    float displaced_point_indices[2];
    float score;
    float image_coord[2];
    float pose_score;
    float not_overlapped_scores;
    char part[100];
    char source_part[100];
    char dest_part[100];
    int thickness = 2;
    int radiusCircle = 5;
    int out_data_index=0;
    cv::Scalar colorCircle(255,255,255);
    cv::Scalar colorLine(255, 255, 0);

    float scale = float(camera_frame.cols) / float(tget_wid);
    

    for (int a = 0; a < NUM_KEYPOINTS; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < arr_size; c++,out_data_index++)
            {
                arr_heatmap_temp[a][b][c] = sigmoid(out_data[headmap_id][out_data_index]);
            }
        }
    }
    //transpose
    for (int a = 0; a < arr_size; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < NUM_KEYPOINTS; c++)
            {
                arr_heatmap[a][b][c] =  arr_heatmap_temp[c][a][b];
            }
        }
    }

    out_data_index=0;
    for (int a = 0; a < NUM_KEYPOINTS*2; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < arr_size; c++,out_data_index++)
            {
                arr_offset_temp[a][b][c] = out_data[offset_id][out_data_index];
            }
        }
    }
    //transpose
    for (int a = 0; a < arr_size; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < NUM_KEYPOINTS*2; c++)
            {
                arr_offset[a][b][c] =  arr_offset_temp[c][a][b];
            }
        }
    }

    out_data_index=0;
    for (int a = 0; a < NUM_CHAIN*2; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < arr_size; c++,out_data_index++)
            {
                arr_fwd_temp[a][b][c] = out_data[fwd_id][out_data_index];
            }
        }
    }
    //transpose
    for (int a = 0; a < arr_size; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < NUM_CHAIN*2; c++)
            {
                arr_fwd[a][b][c] =  arr_fwd_temp[c][a][b];
            }
        }
    }

    out_data_index=0;
    for (int a = 0; a < NUM_CHAIN*2; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < arr_size; c++,out_data_index++)
            {
                arr_bwd_temp[a][b][c] = out_data[bwd_id][out_data_index];
            }
        }
    }
    //transpose
    for (int a = 0; a < arr_size; a++)
    {
        for (int b = 0; b < arr_size; b++)
        {
            for (int c = 0; c < NUM_CHAIN*2; c++)
            {
                arr_bwd[a][b][c] =  arr_bwd_temp[c][a][b];
            }
        }
    }

    for (int  c = 0; c < NUM_KEYPOINTS; c++)
    {
        int part_num_key=0;
        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                if(arr_heatmap[a][b][c] < score_threshold) kp_scores[a][b] = 0;
                else 
                {
                    kp_scores[a][b] = arr_heatmap[a][b][c];
                }
            }
        }

        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                max_heat = kp_scores[a][b];
                square_x1 = a - LOCAL_MAXIMUM_RADIUS;
                square_x2 = a + LOCAL_MAXIMUM_RADIUS;
                square_y1 = b - LOCAL_MAXIMUM_RADIUS;
                square_y2 = b + LOCAL_MAXIMUM_RADIUS;
                if(square_x1 < 0) square_x1 = 0;
                if(square_y1 < 0) square_y1 = 0;
                if(square_x2 >= arr_size) square_x2 = arr_size - 1;
                if(square_y2 >= arr_size) square_y2 = arr_size - 1;

                for(int x = square_x1; x <= square_x2; x++)
                {
                    for(int y = square_y1; y <= square_y2; y++)
                    {
                        if(max_heat < kp_scores[x][y]) max_heat = kp_scores[x][y];
                    }
                }
                max_vals[a][b] = max_heat;
                if((kp_scores[a][b] > 0) && (kp_scores[a][b] == max_vals[a][b])) max_loc[a][b] = 1;
                else max_loc[a][b] = 0;
            }
        }

        for (int b = 0; b < arr_size; b++)
        {
            for (int a = 0; a < arr_size; a++)
            {
                if(max_loc[a][b] == 1)
                {
                    parts[part_num][0] = kp_scores[a][b];
                    parts[part_num][1] = c;   // keypoint_id
                    parts[part_num][2] = b;
                    parts[part_num][3] = a;
                    part_num++;
                    part_num_key++;
                }
            }
        }
        
        //sort part base on score: high -> low
        float part_temp[4];
        bool finish_convert = true;
        while(1)
        {
            for (int i = 0; i < part_num_key; i++)
            {
                if((i < part_num_key - 1) && (parts[i][0] < parts[i + 1][0]))
                {
                    part_temp[0] = parts[i+1][0];
                    part_temp[1] = parts[i+1][1];
                    part_temp[2] = parts[i+1][2];
                    part_temp[3] = parts[i+1][3];
                    parts[i+1][0] = parts[i][0];
                    parts[i+1][1] = parts[i][1];
                    parts[i+1][2] = parts[i][2];
                    parts[i+1][3] = parts[i][3];
                    parts[i][0] = part_temp[0];
                    parts[i][1] = part_temp[1];
                    parts[i][2] = part_temp[2];
                    parts[i][3] = part_temp[3];
                    finish_convert = false;
                    break;
                }
                else
                {
                    finish_convert = true;
                }
            }
            if(finish_convert == true) break;
        }
    }

    for (int part_index = 0; part_index < part_num; part_index++)
    {
        root_score = parts[part_index][0];
        root_id = int(parts[part_index][1]);
        root_coord[1] = parts[part_index][2];
        root_coord[0] = parts[part_index][3];
        squared_nms_radius = pow(nms_radius,2);

        root_image_coords[0] = root_coord[0] * float(stride) + arr_offset[int(root_coord[0])][int(root_coord[1])][root_id]; 
        root_image_coords[1] = root_coord[1] * float(stride) + arr_offset[int(root_coord[0])][int(root_coord[1])][root_id + NUM_KEYPOINTS]; 

        if(pose_count != 0)
        {
            float coord[pose_count][2];
            float coord_square[pose_count];
            bool skip_flag = false;
            for(int pose_id = 0; pose_id < pose_count; pose_id++)
            {
                coord[pose_id][0] = pose_keypoint_coords[pose_id][root_id][0];
                coord[pose_id][1] = pose_keypoint_coords[pose_id][root_id][1];
                coord[pose_id][0] = coord[pose_id][0] - root_image_coords[0];
                coord[pose_id][1] = coord[pose_id][1] - root_image_coords[1];
                coord[pose_id][0] = coord[pose_id][0] * coord[pose_id][0];
                coord[pose_id][1] = coord[pose_id][1] * coord[pose_id][1];
                coord_square[pose_id] = coord[pose_id][0] + coord[pose_id][1];
                if(coord_square[pose_id] <= squared_nms_radius) skip_flag = true;
            }
            if(skip_flag == true) continue;
        }
        
        for(int id = 0; id < NUM_KEYPOINTS; id++)
        {
            instance_keypoint_scores[id] = 0;
        }
        instance_keypoint_scores[root_id] = root_score;
        instance_keypoint_coords[root_id][0] = root_image_coords[0];
        instance_keypoint_coords[root_id][1] = root_image_coords[1];

        for(int edge = num_edges-1; edge >= 0; edge--)
        {
            strcpy(source_part,label_chain_map[edge+NUM_CHAIN].c_str());
            strcpy(dest_part,label_chain_map[edge].c_str());
            
            for(int part_id=0; part_id < NUM_KEYPOINTS; part_id++)
            {
                if(strcmp(source_part,label_file_map[part_id].c_str()) == 0) source_keypoint_id = part_id;
                if(strcmp(dest_part,label_file_map[part_id].c_str()) == 0) target_keypoint_id = part_id;
            }
            
            //need to set source_keypoint_id, target_keypoint_id when edge = 15 because the error https://stackoverflow.com/questions/4198675/string-is-not-printing-without-new-line-character-in-c
            if(edge == 15)
            {
                source_keypoint_id = 16;
                target_keypoint_id = 14;
            }
            
            if((instance_keypoint_scores[source_keypoint_id] > 0) && (instance_keypoint_scores[target_keypoint_id] == 0))
            {
                source_keypoint_indices[0] = float(int(instance_keypoint_coords[source_keypoint_id][0] / float(stride) + 0.5));
                source_keypoint_indices[1] = float(int(instance_keypoint_coords[source_keypoint_id][1] / float(stride) + 0.5));
                if(source_keypoint_indices[0] < 0) source_keypoint_indices[0] = 0;
                else if(source_keypoint_indices[0] > ( arr_size - 1)) source_keypoint_indices[0] = arr_size - 1;
                if(source_keypoint_indices[1] < 0) source_keypoint_indices[1] = 0;
                else if(source_keypoint_indices[1] > (arr_size - 1)) source_keypoint_indices[1] = arr_size - 1;
                displaced_point[0] = instance_keypoint_coords[source_keypoint_id][0] + arr_bwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge];
                displaced_point[1] = instance_keypoint_coords[source_keypoint_id][1] + arr_bwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge+NUM_CHAIN];

                displaced_point_indices[0] = float(int(displaced_point[0] / float(stride) + 0.5));
                displaced_point_indices[1] = float(int(displaced_point[1] / float(stride) + 0.5));
                if(displaced_point_indices[0] < 0) displaced_point_indices[0] = 0;
                else if(displaced_point_indices[0] > (arr_size - 1)) displaced_point_indices[0] = arr_size - 1;
                if(displaced_point_indices[1] < 0) displaced_point_indices[1] = 0;
                else if(displaced_point_indices[1] > (arr_size - 1)) displaced_point_indices[1] = arr_size - 1;
                score = arr_heatmap[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id];
                image_coord[0] = displaced_point_indices[0] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id]; 
                image_coord[1] = displaced_point_indices[1] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id + NUM_KEYPOINTS];
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords[target_keypoint_id][0] = image_coord[0];
                instance_keypoint_coords[target_keypoint_id][1] = image_coord[1];
            }
        }

        for(int edge = 0; edge < num_edges; edge++)
        {
            strcpy(source_part,label_chain_map[edge].c_str());
            strcpy(dest_part,label_chain_map[edge+NUM_CHAIN].c_str());
            for(int part_id=0; part_id < NUM_KEYPOINTS; part_id++)
            {
                if(strcmp(source_part,label_file_map[part_id].c_str()) == 0) source_keypoint_id = part_id;
                if(strcmp(dest_part,label_file_map[part_id].c_str()) == 0) target_keypoint_id = part_id;
            }
            
            //need to set source_keypoint_id, target_keypoint_id when edge = 15 because the error https://stackoverflow.com/questions/4198675/string-is-not-printing-without-new-line-character-in-c
            if(edge == 15)
            {
                source_keypoint_id = 14;
                target_keypoint_id = 16;
            }
            
            if((instance_keypoint_scores[source_keypoint_id] > 0) && (instance_keypoint_scores[target_keypoint_id] == 0))
            {
                source_keypoint_indices[0] = float(int(instance_keypoint_coords[source_keypoint_id][0] / float(stride) + 0.5));
                source_keypoint_indices[1] = float(int(instance_keypoint_coords[source_keypoint_id][1] / float(stride) + 0.5));
                if(source_keypoint_indices[0] < 0) source_keypoint_indices[0] = 0;
                else if(source_keypoint_indices[0] > ( arr_size - 1)) source_keypoint_indices[0] = arr_size - 1;
                if(source_keypoint_indices[1] < 0) source_keypoint_indices[1] = 0;
                else if(source_keypoint_indices[1] > (arr_size - 1)) source_keypoint_indices[1] = arr_size - 1;
                displaced_point[0] = instance_keypoint_coords[source_keypoint_id][0] + arr_fwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge];
                displaced_point[1] = instance_keypoint_coords[source_keypoint_id][1] + arr_fwd[int(source_keypoint_indices[0])][int(source_keypoint_indices[1])][edge+NUM_CHAIN];

                displaced_point_indices[0] = float(int(displaced_point[0] / float(stride) + 0.5));
                displaced_point_indices[1] = float(int(displaced_point[1] / float(stride) + 0.5));
                if(displaced_point_indices[0] < 0) displaced_point_indices[0] = 0;
                else if(displaced_point_indices[0] > (arr_size - 1)) displaced_point_indices[0] = arr_size - 1;
                if(displaced_point_indices[1] < 0) displaced_point_indices[1] = 0;
                else if(displaced_point_indices[1] > (arr_size - 1)) displaced_point_indices[1] = arr_size - 1;
                score = arr_heatmap[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id];
                image_coord[0] = displaced_point_indices[0] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id]; 
                image_coord[1] = displaced_point_indices[1] * float(stride) + arr_offset[int(displaced_point_indices[0])][int(displaced_point_indices[1])][target_keypoint_id + NUM_KEYPOINTS];
                instance_keypoint_scores[target_keypoint_id] = score;
                instance_keypoint_coords[target_keypoint_id][0] = image_coord[0];
                instance_keypoint_coords[target_keypoint_id][1] = image_coord[1];
            }
        }

        not_overlapped_scores = 0;
        if(pose_count != 0)
        {
            float coord_cal[pose_count][NUM_KEYPOINTS][2];
            float coord_square_cal[pose_count][NUM_KEYPOINTS];
            bool bigger = false;
            for(int pose_id = 0; pose_id < pose_count; pose_id++)
            {
                for(int key = 0; key < NUM_KEYPOINTS; key++)
                {
                    coord_cal[pose_id][key][0] = pose_keypoint_coords[pose_id][key][0];
                    coord_cal[pose_id][key][1] = pose_keypoint_coords[pose_id][key][1];
                    coord_cal[pose_id][key][0] -= instance_keypoint_coords[key][0];
                    coord_cal[pose_id][key][1] -= instance_keypoint_coords[key][1];
                    coord_cal[pose_id][key][0] = coord_cal[pose_id][key][0] * coord_cal[pose_id][key][0];
                    coord_cal[pose_id][key][1] = coord_cal[pose_id][key][1] * coord_cal[pose_id][key][1];
                    coord_square_cal[pose_id][key] = coord_cal[pose_id][key][0] + coord_cal[pose_id][key][1];
                }
            }
            for(int key = 0; key < NUM_KEYPOINTS; key++)
            {
                for(int pose_id = 0; pose_id < pose_count; pose_id++)
                {
                    if(coord_square_cal[pose_id][key] > squared_nms_radius) bigger = true;
                    else
                    {
                        bigger = false;
                        break;
                    }
                }
                if(bigger == true) not_overlapped_scores += instance_keypoint_scores[key];
            }
        }
        else
        {
            for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
            {
                not_overlapped_scores += instance_keypoint_scores[keypoint_id];
            }
        }
        pose_score = not_overlapped_scores / NUM_KEYPOINTS;

        if(pose_score >= MIN_POSE_SCORE)
        {
            pose_scores[pose_count] = pose_score;
            for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
            {
                pose_keypoint_scores[pose_count][keypoint_id] = instance_keypoint_scores[keypoint_id];
                pose_keypoint_coords[pose_count][keypoint_id][0] = instance_keypoint_coords[keypoint_id][0];
                pose_keypoint_coords[pose_count][keypoint_id][1] = instance_keypoint_coords[keypoint_id][1];
            }
            pose_count += 1;
        }
        if(pose_count >= MAX_POSE_DETECTIONS) break;
    }
    
    if(print_poses_score == 1)
    {
        //disable print result in case run real time with camera
        for(int pose_id=0; pose_id < pose_count; pose_id++)
        {
            if(pose_scores[pose_id] == 0) break;
            printf("\nPose %d, score = %f \n",  pose_id,pose_scores[pose_id]);
            for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
            {
                printf(" Keypoint = %s \n",  label_file_map[keypoint_id].c_str());
                printf("score = %f, coord = [%f %f]\n",  pose_keypoint_scores[pose_id][keypoint_id], pose_keypoint_coords[pose_id][keypoint_id][0]*scale, pose_keypoint_coords[pose_id][keypoint_id][1]*scale);
            }
        }
    }

    for(int pose_id=0; pose_id < pose_count; pose_id++)
    {
        for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
        {
            //####################(  Draw keypoints  )#########################
            if(pose_keypoint_scores[pose_id][keypoint_id] >= MIN_POSE_SCORE)
            {
                cv::Point centerCircle(pose_keypoint_coords[pose_id][keypoint_id][1]*scale,pose_keypoint_coords[pose_id][keypoint_id][0]*scale);
                cv::circle(camera_frame, centerCircle, radiusCircle, colorCircle, thickness);
            }
        }
        for(int keypoint_id = 0; keypoint_id < NUM_KEYPOINTS; keypoint_id++)
        {
            //####################(  Draw skeleton  )#########################
            if((keypoint_id == 15) && (pose_keypoint_scores[pose_id][15] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][13] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][15][1]*scale,pose_keypoint_coords[pose_id][15][0]*scale), p2(pose_keypoint_coords[pose_id][13][1]*scale,pose_keypoint_coords[pose_id][13][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 11) && (pose_keypoint_scores[pose_id][11] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][13] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][11][1]*scale,pose_keypoint_coords[pose_id][11][0]*scale), p2(pose_keypoint_coords[pose_id][13][1]*scale,pose_keypoint_coords[pose_id][13][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 11) && (pose_keypoint_scores[pose_id][11] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][12] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][11][1]*scale,pose_keypoint_coords[pose_id][11][0]*scale), p2(pose_keypoint_coords[pose_id][12][1]*scale,pose_keypoint_coords[pose_id][12][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 11) && (pose_keypoint_scores[pose_id][11] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][5] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][11][1]*scale,pose_keypoint_coords[pose_id][11][0]*scale), p2(pose_keypoint_coords[pose_id][5][1]*scale,pose_keypoint_coords[pose_id][5][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 5) && (pose_keypoint_scores[pose_id][6] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][5] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][6][1]*scale,pose_keypoint_coords[pose_id][6][0]*scale), p2(pose_keypoint_coords[pose_id][5][1]*scale,pose_keypoint_coords[pose_id][5][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 7) && (pose_keypoint_scores[pose_id][7] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][9] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][7][1]*scale,pose_keypoint_coords[pose_id][7][0]*scale), p2(pose_keypoint_coords[pose_id][9][1]*scale,pose_keypoint_coords[pose_id][9][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 7) && (pose_keypoint_scores[pose_id][7] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][5] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][7][1]*scale,pose_keypoint_coords[pose_id][7][0]*scale), p2(pose_keypoint_coords[pose_id][5][1]*scale,pose_keypoint_coords[pose_id][5][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            
            if((keypoint_id == 16) && (pose_keypoint_scores[pose_id][16] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][14] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][16][1]*scale,pose_keypoint_coords[pose_id][16][0]*scale), p2(pose_keypoint_coords[pose_id][14][1]*scale,pose_keypoint_coords[pose_id][14][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 12) && (pose_keypoint_scores[pose_id][12] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][14] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][12][1]*scale,pose_keypoint_coords[pose_id][12][0]*scale), p2(pose_keypoint_coords[pose_id][14][1]*scale,pose_keypoint_coords[pose_id][14][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 12) && (pose_keypoint_scores[pose_id][12] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][6] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][12][1]*scale,pose_keypoint_coords[pose_id][12][0]*scale), p2(pose_keypoint_coords[pose_id][6][1]*scale,pose_keypoint_coords[pose_id][6][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 8) && (pose_keypoint_scores[pose_id][8] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][6] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][8][1]*scale,pose_keypoint_coords[pose_id][8][0]*scale), p2(pose_keypoint_coords[pose_id][6][1]*scale,pose_keypoint_coords[pose_id][6][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
            if((keypoint_id == 8) && (pose_keypoint_scores[pose_id][8] >= MIN_POSE_SCORE) && (pose_keypoint_scores[pose_id][10] >= MIN_POSE_SCORE))
            {
                cv::Point p1(pose_keypoint_coords[pose_id][8][1]*scale,pose_keypoint_coords[pose_id][8][0]*scale), p2(pose_keypoint_coords[pose_id][10][1]*scale,pose_keypoint_coords[pose_id][10][0]*scale);
                cv::line(camera_frame, p1, p2, colorLine, thickness);
            }
        }
    }
    
    video.write(camera_frame);

    return ret;
}

void run_model(){
    std::vector<OrtValue *> output_tensor(4);
    output_tensor[0] = NULL;
    output_tensor[1] = NULL;
    output_tensor[2] = NULL;
    output_tensor[3] = NULL;
    int is_tensor;

    // check parameter
    CheckStatus(g_ort->IsTensor(input_tensor[0], &is_tensor));
    assert(is_tensor);

    CheckStatus(g_ort->Run(session, NULL, input_node_names.data(), input_tensor.data(), num_input_nodes, output_node_names.data(), num_output_nodes, output_tensor.data()));
    for (int i = 0; i <= 3; i++)
    {
        CheckStatus(g_ort->IsTensor(output_tensor[i],&is_tensor));
        assert(is_tensor);
        
        // Get pointer to output tensor float values
        g_ort->GetTensorMutableData(output_tensor[i], (void**)&out_data[i]);        
    }
}

int main(int argc, char* argv[])
{
    int ret = 0;
    if(ret = parse_argument(argc, argv)) {
        return ret;
    }

    if(ret = prepare_environment()){
        return ret;
    }
    

    struct timeval time1, time2, time3, time4;
    double duration;

    //prepare camera
    // Create a VideoCapture object and use camera to capture the video
    VideoCapture cap(cam_index);
    // Check if camera opened successfully
    if(!cap.isOpened())
    {
        cout << "Error opening video stream" << endl;
        return -1;
    }
    // Default resolution of the frame is obtained.The default resolution is system dependent.
    int full_size_video = cap.get(CAP_PROP_FRAME_WIDTH) > cap.get(CAP_PROP_FRAME_HEIGHT) ? cap.get(CAP_PROP_FRAME_WIDTH) : cap.get(CAP_PROP_FRAME_HEIGHT);
    // Define the codec and create VideoWriter object.The output is stored in 'cam_output.avi' file.
    VideoWriter video("output/cam_output.avi",VideoWriter::fourcc('M','J','P','G'),10, Size(full_size_video,full_size_video));
    
    // setup ONNX runtime env
    prepare_ONNX_Runtime();
    
    while(1)
    {
        // preprocessing
        gettimeofday(&time1, nullptr);
        preprocess_input(cap);

        // run inference
        gettimeofday(&time2, nullptr);
        run_model();

        // postprocessing
        gettimeofday(&time3, nullptr);
        postprocess(video);
        gettimeofday(&time4, nullptr);

        if(measure_time == 1)
        {
            duration = timedifference_msec(time1,time2);
            printf("preprocessing Time: %.3f msec\n", duration);
            duration = timedifference_msec(time2,time3);
            printf("run model Time: %.3f msec\n", duration);
            duration = timedifference_msec(time3,time4);
            printf("postprocessing Time: %.3f msec\n", duration);
            duration = timedifference_msec(time1,time4);
            printf("total time of 1 fram processing: %.3f msec\n", duration);
            printf("number of frame per second: %.3f fps\n", 1000/duration);
            printf("\n");
        }

    }
    // When everything done, release the video capture and write object
    cap.release();
    video.release();

    return 0;
}
