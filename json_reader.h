#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include "my_macros.h"

std::vector<bool> get_bool_property(std::string file_name = "../models/run1.json");
std::vector<std::string> get_property(std::string file_name = "../models/run1.json");
void json_info(std::string file_name = "../models/run1.json");

// 读取网格信息
void get_mesh(std::string file_name, std::vector<T_NUMBER>& px, std::vector<T_NUMBER>& py, std::vector<T_NUMBER>& pz);
void str2mesh(const std::string& fileContent, size_t gridStart, std::string axis, std::vector<T_NUMBER>& xs);