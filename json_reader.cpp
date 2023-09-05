#include "json_reader.h"
#include "my_macros.h"
std::vector<bool> get_bool_property(std::string file_name){
    // 封装了get_property，将输出转化为bool类型
    std::vector<bool> bool_properties;
    std::vector<std::string> materials = get_property(file_name);
    for(const std::string& material : materials){
        if(material == "Copper"){
            bool_properties.push_back(true);
        }else{
            bool_properties.push_back(false);
        }
    }
    return bool_properties;
}

std::vector<std::string> get_property(std::string file_name){
    // 读取json文件，并返回一个vector<string> 表示每个零件的材料
    std::vector<std::string> materials;
    std::ifstream jsonFile(file_name);
    
    if (!jsonFile.is_open()) {
        // std::cerr << "无法打开文件." << std::endl;
        // return;
        throw std::runtime_error("无法打开文件.");
    }
    
    // 读取整个文件内容
    std::string fileContent((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
    
    jsonFile.close();
    
    // 去除空格和换行，使解析更容易
    fileContent.erase(std::remove_if(fileContent.begin(), fileContent.end(), ::isspace), fileContent.end());
    
    // 寻找 "solids" 字段
    std::size_t solidsStart = fileContent.find("\"solids\":[{");
    if (solidsStart == std::string::npos) {
        // std::cerr << "找不到 solids 字段." << std::endl;
        // return;
        throw std::runtime_error("找不到 solids 字段.");
    }
    
    // 寻找 "material" 属性
    std::size_t materialStart = fileContent.find("\"material\":\"", solidsStart);
    int cnt = 0;
    while (materialStart != std::string::npos) {
        materialStart += 12; // 跳过 "\"material\":\"" 的长度
        
        std::size_t materialEnd = fileContent.find("\"", materialStart);
        std::string material = fileContent.substr(materialStart, materialEnd - materialStart);
        // std::cout << "Material: " << material << std::endl;
        materials.push_back(material);
        // 继续寻找下一个 "material" 属性
        materialStart = fileContent.find("\"material\":\"", materialEnd);
        cnt++;
    }
    return materials;
}

void json_info(std::string file_name){
    std::unordered_map<std::string, int> Material_cnt; // first: material name  second: count
    std::vector<std::string> materials = get_property(file_name);
    for(const std::string& material : materials){
        auto it = Material_cnt.find(material);
        if(it != Material_cnt.end()){
            it->second++;
        }else{
            Material_cnt[material] = 1;
        }
    }
    for(const auto& pair : Material_cnt){
        std::cout << pair.first << " " << pair.second << std::endl;
    }
}

void get_mesh(std::string file_name, std::vector<T_NUMBER>& px, std::vector<T_NUMBER>& py, std::vector<T_NUMBER>& pz){
    std::ifstream jsonFile(file_name);
    
    if (!jsonFile.is_open()) {
        // std::cerr << "无法打开文件." << std::endl;
        // return;
        throw std::runtime_error("无法打开文件.");
    }
    
    // 读取整个文件内容
    std::string fileContent((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
    
    jsonFile.close();
    
    // 去除空格和换行，使解析更容易
    fileContent.erase(std::remove_if(fileContent.begin(), fileContent.end(), ::isspace), fileContent.end());
    
    // 寻找"\"grid\":{"字段
    size_t gridStart = fileContent.find("\"grid\":{");
    if (gridStart == std::string::npos) {
        throw std::runtime_error("找不到grid数据.");
    }

    str2mesh(fileContent, gridStart, "\"xs\":[", px);
    str2mesh(fileContent, gridStart, "\"ys\":[", py);
    str2mesh(fileContent, gridStart, "\"zs\":[", pz);
    #ifdef ENABLE_PRINT
        std::cout << "mesh info: "<< std::endl;
        std::cout << "xs: " << px.size() << std::endl;
        std::cout << "ys: " << py.size() << std::endl;
        std::cout << "zs: " << pz.size() << std::endl;
    #endif
}

void str2mesh(const std::string& fileContent, size_t gridStart, std::string axis, std::vector<T_NUMBER>& xs){
    // axis = "\"xs\":["
    size_t xsStart = fileContent.find(axis, gridStart);
    size_t xsEnd = fileContent.find("]", xsStart);
    if (xsStart == std::string::npos || xsEnd == std::string::npos) {
        throw std::runtime_error("找不到" + axis + "字段.");
    }
    xsStart += 6;
    std::string xsData = fileContent.substr(xsStart, xsEnd - xsStart);

    // 解析 xsData 为 T_NUMBER 数组
    size_t pos = 0;
    while (pos < xsData.length()) {
        size_t commaPos = xsData.find(",", pos);
        if (commaPos == std::string::npos) {
            commaPos = xsData.length();
        }
        xs.push_back(std::stod(xsData.substr(pos, commaPos - pos)));
        pos = commaPos + 1;
    }
}