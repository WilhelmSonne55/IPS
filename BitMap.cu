#include "BitMap.h"
#include "stdio.h"
#if 0
#include<iostream>
using namespace std;
#endif
//========================================
//  Definition
//========================================.
#define BLOCKSIZE 32 	//1024 = 32*32 thread is a limit
#define MAX_DEPTH 0xFFFF
#define SIXTY_DEGREE 60
#define RGB_SCALE 100
#define HSV_SCALE 0.3

#define BOUND(x,min, max) ((x) > (max) ? (max): ((x) < (min)? (min): (x)))
#define MAXRGB(R,G,B) (R)>(G)?(R>B?R:B):((G)>(B)?(G):(B))
#define MINRGB(R,G,B) (R)<(G)?(R<B?R:B):((G<B)?G:B)

//========================================
//  CUDA Function
//========================================
__global__ void AdjustColorKernel(unsigned short* devPtr, int width, int height, int* RGBvalue)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = tidx + tidy * width;
	
	if(tidx <= width && tidy <= height)
	{		
		devPtr[offset*4 + RED_PIXEL] = BOUND(devPtr[offset*4 + RED_PIXEL] + RGBvalue[RED_PIXEL]*(MAX_DEPTH/RGB_SCALE), 0, MAX_DEPTH);
		devPtr[offset*4 + GREEN_PIXEL] = BOUND(devPtr[offset*4 + GREEN_PIXEL] + RGBvalue[GREEN_PIXEL]*(MAX_DEPTH/RGB_SCALE), 0, MAX_DEPTH);
		devPtr[offset*4 + BLUE_PIXEL] = BOUND(devPtr[offset*4 + BLUE_PIXEL] + RGBvalue[BLUE_PIXEL]*(MAX_DEPTH/RGB_SCALE), 0, MAX_DEPTH);
	}
}

__device__ void ConvertHSVtoRGB(unsigned short* devPtr, int offset, float H, float S, float V)
{
	float R = 0, G = 0, B = 0;
	float h = (int)H/60;
	float f = H/60.0 - h;
	float p = V*(1-S);
	float q = V*(1-f*S);
	float t = V*(1-(1-f)*S);
	
	switch((int)h)
	{
		case 0:
			R = V;
			G = t;
			B = p;
			break;
		case 1:
			R = q;
			G = V;
			B = p;
			break;
		case 2:
			R = p;
			G = V;
			B = t;
			break;
		case 3:
			R = p;
			G = q;
			B = V;
			break;
		case 4:
			R = t;
			G = p;
			B = V;
			break;
		default:
			R = V;
			G = p;
			B = q;
			break;
	}	
	
	devPtr[offset*4] = R*MAX_DEPTH;
	devPtr[offset*4+1] = G*MAX_DEPTH;
	devPtr[offset*4+2] = B*MAX_DEPTH;
}

__global__ void ConvertRGBtoHSV(unsigned short* devPtr, int width, int height, int* HSVvalue)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = tidx + tidy * width;
	
	if(tidx <= width && tidy <= height)
	{	
		float H = 0, S = 0, V = 0;
		float R = devPtr[offset*4];
		float G = devPtr[offset*4 + 1];
		float B = devPtr[offset*4 + 2];
		float max = MAXRGB(R,G,B);
		float min = MINRGB(R,G,B);
		float delta = max - min;
				
		//H
		if(min == max)
			H = 0;
		else if(max == R && G >= B)
			H=(SIXTY_DEGREE*(G-B)/delta);
		else if(max == R && G < B) 
			H=SIXTY_DEGREE*((G-B)/delta + 6);
		else if(max == G)
			H=SIXTY_DEGREE*((B-R)/delta + 2);
		else  //max == B
			H=SIXTY_DEGREE*((R-G)/delta +4);
			
		//Brightness
		V = max/MAX_DEPTH;
		
		//Saturation
		if(max == 0)
		{
			S = 0;
		}
		else
		{
			S = 1 - min/max;
		}
			
		//RED
		if(0 <= H && H < 15)
		{			
			H += 360;
			H = BOUND(H + HSVvalue[HUE_RED_TYPE], 330, 374);
						
			if(H >= 360)
				H -= 360;
			
			S = BOUND(S+HSVvalue[SAT_RED_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_RED_TYPE]*0.01,0,1);
		}
		else if(315 <= H && H <= 360)
		{
			H = BOUND(H+ HSVvalue[HUE_RED_TYPE], 315, 360);
			if(H >= 360)
				H -= 360;
			
			S = BOUND(S+HSVvalue[SAT_RED_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_RED_TYPE]*0.01,0,1);
		}
		else if(15 <= H && H < 45)
		{
			H = BOUND(H+HSVvalue[HUE_ORANGE_TYPE]*HSV_SCALE, 15, 44);
			S = BOUND(S+HSVvalue[SAT_ORANGE_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_ORANGE_TYPE]*0.01,0,1);
		}
		else if(45 <= H && H < 75)
		{
			H = BOUND(H+HSVvalue[HUE_YELLOW_TYPE]*HSV_SCALE, 45, 74);
			S = BOUND(S+HSVvalue[SAT_YELLOW_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_YELLOW_TYPE]*0.01,0,1);
		}
		else if(75 <= H && H < 165)
		{
			H = BOUND(H + HSVvalue[HUE_GREEN_TYPE]*HSV_SCALE, 75, 164);	
			S = BOUND(S+HSVvalue[SAT_GREEN_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_GREEN_TYPE]*0.01,0,1);
		}
		else if(165 <= H && H < 225)
		{
			H = BOUND(H+HSVvalue[HUE_CYAN_TYPE]*HSV_SCALE, 165, 224);
			S = BOUND(S+HSVvalue[SAT_CYAN_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_CYAN_TYPE]*0.01,0,1);
		}
		else if(225 <= H && H < 255)
		{
			H=BOUND(H+HSVvalue[HUE_BLUE_TYPE]*HSV_SCALE, 225, 254);	
			S = BOUND(S+HSVvalue[SAT_BLUE_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_BLUE_TYPE]*0.01,0,1);
		}
		else if(255 <= H && H < 285)
		{
			H=BOUND(H+HSVvalue[HUE_VIOLET_TYPE]*HSV_SCALE, 255, 284);
			S = BOUND(S+HSVvalue[SAT_VIOLET_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_VIOLET_TYPE]*0.01,0,1);
		}
		else if(285 <= H && H < 315)
		{	 
			H=BOUND(H+HSVvalue[HUE_MAGENTA_TYPE]*HSV_SCALE, 285, 314);
			S = BOUND(S+HSVvalue[SAT_MAGENTA_TYPE]*0.01, 0, 1);
			V = BOUND(V+(float)HSVvalue[BRI_MAGENTA_TYPE]*0.01,0,1);
		}
						
		ConvertHSVtoRGB(devPtr, offset, H, S, V);
	}
}

//========================================
//  Function
//========================================
BitMap::BitMap(unsigned short* _data, int _width, int _height, int _channel)
{
	data = _data;
	width = _width;
	height = _height;
	channel = _channel;
	size = width*height*channel*sizeof(unsigned short);	
	proImage = new unsigned short[size];
	memcpy(proImage , data, size);
}

void BitMap::refresh()
{
	memcpy(proImage, data, size);
}

void Processor::process(BitMap* bitmap, COLOR_ITEMS uiSlider)
{
	if(memcmp(&slider, &uiSlider, sizeof(COLOR_ITEMS)) == 0)
	{
		return;
	}
	
	//to copy the original image to new one for processing
	bitmap->refresh();
	
	//RGB 
	AdjustColor(bitmap, uiSlider.RGBvalue);

	//HSY
	AdjustHSV(bitmap,  uiSlider.HSVvalue);
	
	memcpy(&slider, &uiSlider, sizeof(COLOR_ITEMS));
}

void Processor::AdjustColor(BitMap* bitmap, int* RGBvalue)
{
	unsigned short* devPtr;
	int* devRGBvalue;
	int height = bitmap->get_height();
	int width = bitmap->get_width();
	int size = bitmap->get_size();
		
	cudaMalloc((void**)&devRGBvalue, sizeof(int)*MAX_PIXEL);
	cudaMemcpy( devRGBvalue, RGBvalue, sizeof(int)*MAX_PIXEL ,cudaMemcpyHostToDevice );
	
	cudaMalloc( (void**)&devPtr, size );
	cudaMemcpy( devPtr, bitmap->get_Image(), size,cudaMemcpyHostToDevice );

	int bx = (width + BLOCKSIZE - 1)/BLOCKSIZE;
	int by = (height + BLOCKSIZE - 1)/BLOCKSIZE;
	dim3 gridSize(bx, by);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
		
	AdjustColorKernel<<<gridSize, blockSize>>>(devPtr, width, height, devRGBvalue);
	
	cudaMemcpy( bitmap->get_Image(), devPtr, size,cudaMemcpyDeviceToHost );
	cudaFree(devRGBvalue);
	cudaFree(devPtr);
}

static bool check = 0;

void CPUHSVtoRGB(unsigned short* devPtr, float H, float S, float V)
{
	float R, G, B = 0;
	float h = (int)H/60;
	float f = H/60.0 - h;
	float p = V*(1-S);
	float q = V*(1-f*S);
	float t = V*(1-(1-f)*S);
	
	switch((int)h)
	{
		case 0:
			R = V;
			G = t;
			B = p;
			break;
		case 1:
			R = q;
			G = V;
			B = p;
			break;
		case 2:
			R = p;
			G = V;
			B = t;
			break;
		case 3:
			R = p;
			G = q;
			B = V;
			break;
		case 4:
			R = t;
			G = p;
			B = V;
			break;
		default:
			R = V;
			G = p;
			B = q;
			break;
	}
		if(check)
		printf(" =>h:%d, H:%f, S:%f, V:%f f:%f, R:%f, G:%f, B:%f \n", (int)h, H,S, V, f,R*255, G*255, B*255);

			check = false;

		devPtr[0] = R*MAX_DEPTH;
		devPtr[1] = G*MAX_DEPTH;
		devPtr[2] = B*MAX_DEPTH;	
		//cout<<"R:"<<R<<" G:"<<G<<" B:"<<B<<endl;
}

void CPUConvertRGBtoHSV(unsigned short* devPtr, int width, int height, int* HSVvalue)
{
		float H = 0, S = 0, V = 0;

		for(int y = 0; y< height; y++)
		{
			for(int x= 0; x< width; x++)
			{
				int offset = y*width+x;
				float R = devPtr[offset*4];
				float G = devPtr[offset*4 + 1];
				float B = devPtr[offset*4 + 2];
				float max = MAXRGB(R,G,B);
				float min = MINRGB(R,G,B);
				float delta = max - min;
				
				//H
				if(min == max)
					H = 0;
				else if(max == R && G >= B)
					H=(SIXTY_DEGREE*(G-B)/delta);
				else if(max == R && G < B) 
					H=SIXTY_DEGREE*((G-B)/delta + 6);
				else if(max == G)
					H=SIXTY_DEGREE*((B-R)/delta + 2);
				else  //max == B
					H=SIXTY_DEGREE*((R-G)/delta +4);				
					
				//Brightness
				V = max/MAX_DEPTH;
				
				//Saturation
				if(max == 0)
				{
					S = 0;
				}
				else
				{
					S = 1 - min/max;
				}
					
				//RED
				if(0 <= H && H < 15)
				{			
					H += 360;
					H = BOUND(H + HSVvalue[HUE_RED_TYPE], 330, 374);
								
					if(H >= 360)
						H -= 360;
					
					S = BOUND(S+HSVvalue[SAT_RED_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_RED_TYPE]*0.01,0,1);
				}
				else if(315 <= H && H <= 360)
				{
					H = BOUND(H+ HSVvalue[HUE_RED_TYPE], 315, 360);
					if(H >= 360)
						H -= 360;
					
					S = BOUND(S+HSVvalue[SAT_RED_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_RED_TYPE]*0.01,0,1);
				}
				else if(15 <= H && H < 45)
				{
					H = BOUND(H+HSVvalue[HUE_ORANGE_TYPE]*HSV_SCALE, 15, 44);
					S = BOUND(S+HSVvalue[SAT_ORANGE_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_ORANGE_TYPE]*0.01,0,1);
				}
				else if(45 <= H && H < 75)
				{
					H = BOUND(H+HSVvalue[HUE_YELLOW_TYPE]*HSV_SCALE, 45, 74);
					S = BOUND(S+HSVvalue[SAT_YELLOW_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_YELLOW_TYPE]*0.01,0,1);
				}
				else if(75 <= H && H < 165)
				{
					H = BOUND(H + HSVvalue[HUE_GREEN_TYPE]*HSV_SCALE, 75, 164);	
					S = BOUND(S+HSVvalue[SAT_GREEN_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_GREEN_TYPE]*0.01,0,1);
				}
				else if(165 <= H && H < 225)
				{
					H = BOUND(H+HSVvalue[HUE_CYAN_TYPE]*HSV_SCALE, 165, 224);
					S = BOUND(S+HSVvalue[SAT_CYAN_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_CYAN_TYPE]*0.01,0,1);
				}
				else if(225 <= H && H < 255)
				{
					H=BOUND(H+HSVvalue[HUE_BLUE_TYPE]*HSV_SCALE, 225, 254);	
					S = BOUND(S+HSVvalue[SAT_BLUE_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_BLUE_TYPE]*0.01,0,1);
				}
				else if(255 <= H && H < 285)
				{
					H=BOUND(H+HSVvalue[HUE_VIOLET_TYPE]*HSV_SCALE, 255, 284);
					S = BOUND(S+HSVvalue[SAT_VIOLET_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_VIOLET_TYPE]*0.01,0,1);
				}
				else if(285 <= H && H < 315)
				{	 
					H=BOUND(H+HSVvalue[HUE_MAGENTA_TYPE]*HSV_SCALE, 285, 314);
					S = BOUND(S+HSVvalue[SAT_MAGENTA_TYPE]*0.01, 0, 1);
					V = BOUND(V+(float)HSVvalue[BRI_MAGENTA_TYPE]*0.01,0,1);
				}
				
				if(x == 0 && y == 0)
				{
					printf(" H:%f, S:%f, V:%f, R:%f, G:%f, B:%f \n", H, S, V,R*255/MAX_DEPTH, G*255/MAX_DEPTH, B*255/MAX_DEPTH);
					check = true;
				}

				CPUHSVtoRGB(&devPtr[offset*4], H, S, V);
			}
		}
}


void Processor::AdjustHSV(BitMap* bitmap, int* HSVvalue)
{
	unsigned short* devPtr;
	int* devHSVvalue;
	int height = bitmap->get_height();
	int width = bitmap->get_width();
	int size = bitmap->get_size();
	
	//CPUConvertRGBtoHSV(bitmap->get_Image(), width, height, HSVvalue);
	
	#if 1
	cudaMalloc((void**)&devHSVvalue, sizeof(int)*HSV_MAX_TYPE);
	cudaMemcpy( devHSVvalue, HSVvalue, sizeof(int)*HSV_MAX_TYPE, cudaMemcpyHostToDevice );
	
	cudaMalloc( (void**)&devPtr, size );
	cudaMemcpy( devPtr, bitmap->get_Image(), size,cudaMemcpyHostToDevice );

	int bx = (width + BLOCKSIZE - 1)/BLOCKSIZE;
	int by = (height + BLOCKSIZE - 1)/BLOCKSIZE;
	dim3 gridSize(bx, by);
	dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
	
	ConvertRGBtoHSV<<<gridSize, blockSize>>>(devPtr, width, height, devHSVvalue);
	
	cudaMemcpy(bitmap->get_Image(), devPtr, size,cudaMemcpyDeviceToHost );

	cudaFree(devHSVvalue);
	cudaFree(devPtr);	
	#endif
}
