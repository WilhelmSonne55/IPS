#ifndef BITMAP_H
#define BIT_MAP_H

#ifdef DLL_IMPORT
#define DLL __declspec(dllimport)
#else
#define DLL __declspec(dllexport)
#endif

//========================================
//  Function
//========================================
typedef enum DLL 
{
	RED_PIXEL,
	GREEN_PIXEL,
	BLUE_PIXEL,
	MAX_PIXEL
} ColorPixel;

typedef enum DLL
{
	HUE_RED_TYPE,         // 0 DEGREE
	SAT_RED_TYPE,
	BRI_RED_TYPE,

	HUE_ORANGE_TYPE,  //30 DEGREE
	SAT_ORANGE_TYPE,
	BRI_ORANGE_TYPE,
	
	HUE_YELLOW_TYPE,      //60 DEGREE
	SAT_YELLOW_TYPE,
	BRI_YELLOW_TYPE,
	
	
	HUE_GREEN_TYPE,       //120 DEGREE
	SAT_GREEN_TYPE,
	BRI_GREEN_TYPE,
	
	HUE_CYAN_TYPE,        //180 DEGREE
	SAT_CYAN_TYPE,
	BRI_CYAN_TYPE,
	
	HUE_BLUE_TYPE,        //240 DEGREE
	SAT_BLUE_TYPE,
	BRI_BLUE_TYPE,
	
	HUE_VIOLET_TYPE,      //270 DEGREE
	SAT_VIOLET_TYPE,
	BRI_VIOLET_TYPE,
	
	HUE_MAGENTA_TYPE,     //300 DEGREE
	SAT_MAGENTA_TYPE,
	BRI_MAGENTA_TYPE,
	
	HSV_MAX_TYPE
} HSV_TYPE;

typedef struct{
	int RGBvalue[MAX_PIXEL] = {0};
	int HSVvalue[HSV_MAX_TYPE] = {0};
}COLOR_ITEMS;

typedef struct{
	float* RHistogram;
	float* GHistogram;
	float* BHistogram;
	float* YHistogram;
}Histogram_Type;

class DLL BitMap
{
	public:
	
	BitMap(unsigned short* _data, int _width, int _height, int _channel, int _depth);
    unsigned short* get_data( void ) const   { return data; }
    unsigned int get_width( void ) const   { return width; }
    unsigned int get_height( void ) const   { return height; }
    unsigned int get_channel( void ) const   { return channel; }
    unsigned int get_size( void ) const   { return size; }
    unsigned int get_depth( void ) const   { return depth; }
	unsigned short* get_Image (void) {return proImage;};
	float* getYhistogram (void) {return histogram.YHistogram; }
	float* getRhistogram (void) {return histogram.RHistogram; }
	float* getGhistogram (void) {return histogram.GHistogram; }
	float* getBhistogram (void) {return histogram.BHistogram; }

	void refresh();
	void updateHistogram();

	private:
	Histogram_Type histogram;
	unsigned short* data;
	unsigned short* proImage;

	unsigned int binSize;
	unsigned int depth;
	unsigned int width;
	unsigned int height;
	unsigned int channel;
	unsigned int size;	
};

class DLL Processor
{
	public:
	void process(BitMap* bitmap, COLOR_ITEMS slider);
	
	void AdjustColor(BitMap* bitmap, int* RGBvalue);
	void AdjustHSV(BitMap* bitmap, int* HSVvalue);
	
	private:
	COLOR_ITEMS slider;
};



#endif