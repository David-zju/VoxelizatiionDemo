#pragma once
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
using std::string;
using std::ofstream;
using std::ifstream;
namespace Ohm_slice{
struct Pixel {
	// order 代表的是材料
	// geom 代表零件id
	// region 为几何填充中patch上的区域标号
    short order, region, geom;
	Pixel(){}
	Pixel(short order1,short region1,short geom1):order(order1),region(region1),geom(geom1){}
};

struct Texture {
    int width, height;
    Texture(size_t width, size_t height, int init, Pixel *buf): buf(buf) {
		// 存储16*16个Pixel的数据
        this->width = width;
        this->height = height;
        data = new Pixel *[height];
        for (int i = 0; i < height; i ++) {
            data[i] = buf + i * width;
            for (int j = 0; j < width; j ++) {
                data[i][j].order = init;
                data[i][j].region = -1;
                data[i][j].geom = -1;
            }
        }
    }

    ~Texture() {
        delete data;
    }
    Pixel *&operator[] (int i) {
        return data[i];
    }
    Pixel* GetPixel(size_t h,size_t w){
        return &data[h][w];
    }
private:
    Pixel **data;
    Pixel *buf;
    Texture() { }

};

struct Pixels {
    int width, height, thickness;
    Pixels(size_t thickness, size_t width, size_t height, int init):
            width(width), height(height), thickness(thickness) {
        buf = new Pixel[thickness * width * height];
        data = new Texture *[thickness];
        for (int i = 0; i < thickness; i ++) {
            data[i] = new Texture(width, height, init, buf + i * width * height);
        }
    }

    Texture &operator [](int i) {
        return *data[i];
    }

    void Dump(string file) {
        ofstream fn(file);
        fn.write((char *) buf, sizeof(Pixel) * thickness * width * height);
    }

    void Load(string file) {
        ifstream fn(file,std::ios::binary);
        if(!fn){
            std::cout<<"fail to open\n"<<std::endl;
            return;
        }
        //fn.read((char*)buf,sizeof(Pixel) * thickness * width *height);
        for(int i=0;i<thickness;i++){
            fn.read((char*)(buf+i * width * height),sizeof(Pixel) * width * height);
        }
    }

    ~Pixels() {
        for (int i = 0; i < thickness; i ++) {
            delete data[i];
        }
        delete data;
        delete buf;
    }
	Texture** GetData(){
		return data;
	}
	
private:
    Texture **data;
    Pixel *buf;
    Pixels() { }
};
}