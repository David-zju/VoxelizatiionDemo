#pragma once
#include "Cell.h"
#include <fstream>

void saveClashLstToFile(const Ohm_slice::Cell& cell, const std::string& filename, int x, int y, int z) {
    std::ofstream outFile(filename, std::ios::app); // 以追加模式打开文件
    if (outFile.is_open()) {
        outFile << "CellStart " << x << " " << y << " " << z << std::endl; // 标识符和 xyz 值
        for (short value : cell.clash_lst) {
            outFile << value << " "; // 将值写入文件，用空格分隔
        }
        outFile << std::endl;
        outFile << "CellEnd" << std::endl; // 标识符
        outFile.close();
    } else {
        std::cout << "Failed to open file for writing." << std::endl;
    }
}