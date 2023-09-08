# Voxelization Demo

#### 编译说明
- 本项目使用cmake编译，配置内容于CMakeList.txt中
- 本项目依赖库已在目录下，直接编译即可，无需配置环境，建议在Linux下进行编译。
```bash
# 依次执行以下指令
mkdir build
cd build
cmake ..
make
```
#### 文件读取说明
- 需要在根目录下创建models文件夹，将stl以及对应的文件放入其中，然后在`main`函数中修改变量`stl_file`和`json_file`。

#### 程序功能说明
- 程序会自动检测当前环境支持的线程数，并多线程执行，线程数存储在`main`函数中的`numThreads` 变量中。
##### 预处理
> 读取stl，json文件，并存储在相应的数据结构中

- 零件信息存储在Entity对象中，网格信息存储在Cell对象中
- `stl_reader.h`用的是第三方开源库 https://sreiter.github.io/stl_reader/index.html, 通过`load_entities`读取stl文件信息
- `Entity`的`is_metal` 属性由 `get_bool_property`获取，其中`get_property`读取json中的零件的材料名称，返`std::vector<std::string>`，而我们测试的run1中金属只有Copper，因此代码中**只捕捉了属性为Copper的零件**，**实际测试时需要修改**
- `Entity`的成员函数`build_BVH()`会将每个零件的三角形以BVH树的形式重组，加速求交。

##### 获得与Cell相交的零件
- 程序会多线程地通过box-triangle求交获取与cell有交的金属零件（我们本地的切片文件没有正确的geom，因此需要这样处理）
- `InitalMultiThreads()`的最后一个变量是函数指针，传入需要多线程执行的函数
- 这里判断box和三角形求交应用的是AST算法（分离轴算法，应该是我查到最快的）
  - https://zhuanlan.zhihu.com/p/138293968
  - https://cniter.github.io/posts/ff29de94.html
- 最后会print出总共和金属零件相交的cell总数以及这一步执行的时间
##### 获得Dexel信息
- 多线程地通过光线投射方法，对上一步存在有金属零件clash的cell进行处理
- 对于每一个处理的cell，使用`refine_to_dexels`创建若干个dexels，也即光线的出发点（仅创建单方向xoy平面朝向z的光线）。再通过`raycast_dexel`函数，让这些光线和上一步记录和cell相交零件进行求交，获得若干个interval，存储在dexel的`scan_lst`变量中。
- 这里还有两个`refine_to_voxels` `raycast_voxel` 函数，它们的处理过程与上面一样，只不过最后存储为Voxel的形式
