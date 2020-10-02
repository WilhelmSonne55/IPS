

CXX = cl.exe 
CC = gcc -m64
NVCC = nvcc

#---------------------
## Sources code
#---------------------
#main file
SOURCES += main.cpp 

#imgui wrapper
SOURCES += ./imgui/imgui_impl_glfw.cpp ./imgui/imgui_impl_opengl3.cpp

#imgui
SOURCES += ./imgui/imgui.cpp ./imgui_demo.cpp ./imgui_draw.cpp ./imgui_widgets.cpp

#CUDA
CUDASOURCES += BitMap.cu

# Using OpenGL loader: gl3w [default]
SOURCES += ./GL/gl3w.c

OBJS = $(addsuffix .obj, $(basename $(notdir $(SOURCES))))
DLL = $(addsuffix .dll, $(basename $(notdir $(CUDASOURCES))))
#---------------------
## Library
#---------------------
LIBS += glfw3.lib gdi32.lib opengl32.lib shell32.lib BitMap.lib

#---------------------
## FLAGS
#---------------------
CXXFLAGS = -I./include/ -I./imgui/ -DIMGUI_IMPL_OPENGL_LOADER_GL3W 
CFLAGS = $(CXXFLAGS)
CUDAFLAGS = --shared
#---------------------
## Build Files
#---------------------
#main.cpp
%.obj:%.cpp 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
#imgui/	
%.obj:imgui/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
#GL/	using c
%.obj:GL/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

#---------------------
## Commands
#---------------------	
all: Proj 
	@echo Build Complete

Proj: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)  $(LDFLAGS) $(LIBS) 
	
dll:$(CUDASOURCES)
	$(NVCC) -o $(DLL) $(CUDAFLAGS) $(CUDASOURCES)
	
clean: 
	del *.obj *.exe