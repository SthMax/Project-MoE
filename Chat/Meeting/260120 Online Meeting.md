# VLM架构相关探讨会议记录
## 一、VLM架构工具与使用
### （一）工具优势
- Mac端使用豆包记录会议，与系统适配性好，可随时调出执行各类任务，操作便捷性优于其他客户端。
- 豆包模型响应高效，通过划线提问的方式能显著提升问题解决效率。

### （二）代码辅助工具
- 可借助Cloud、Copilot、Claude、GPT、Gemini等工具解读代码。
- Claude存在请求消耗过快的问题；Gemini更适用于创意类工作；建议将VLM完整文件夹提供给工具，以提升解读准确性。

## 二、VLM架构工作机制
### （一）scheduler与executor
- scheduler的核心功能是将任务转化为`scheduler output`，该输出包含类似JSON格式的结构化信息，随后传递给executor执行后续流程。

### （二）GPU worker与GPU model Runner
- **GPU worker**：可理解为多进程管理容器，具备调度多个GPU model Runner的能力，内部包含GPU CACHE管理、GPU内存相关配置等属性。
- **GPU model Runner**：相当于直接对接GPU的执行器，核心工作流程如下：
  1. 基于`scheduler output`中的`number scheduled tokens`确定每个请求的TOKEN数量，通过`KV cache block table`定位KV块在GPU的存储位置，进而构造输入Tensor。
  2. 标记TOKEN状态（prefill或decode）。
  3. 通过`self CUDA graph manager`将数据传输至GPU。
  4. 调用`transformer forward`函数，结合Moe layers执行计算，过程中会触发TOPK调用。
- 关键特性：scheduler不区分prefill和decode操作，最终由GPU model Runner决定TOKEN的处理类型，且支持`chunked prefill`模式。

## 三、并行机制讨论
### （一）expert parallel（专家并行）
- 核心逻辑：将不同的expert分配至不同GPU，适用于Moe架构。
- 优势：Moe训练时天然具备多个独立矩阵（物理上独立存在），无需像非Moe模型那样通过identity Matrix拆分矩阵，可直接实现分布式部署。

### （二）Tensor parallel（张量并行）
- 核心逻辑：将模型层的大型矩阵拆分为多个子矩阵，分布到不同GPU并行计算，计算完成后通过identity矩阵组装结果（类似矩阵分块运算）。
- 示例：将矩阵w拆分为w1、w2、w3、w4，矩阵v拆分为v1、v2、v3、v4，分别在不同GPU执行wi×vi运算，最终合并结果。

### （三）pipeline parallel（流水线并行）
- 核心逻辑：按模型层拆分任务，将不同层分配至不同GPU，例如将transformer的1-3层、4-6层、7-9层分别部署在不同GPU。
- 与Tensor parallel的关系：二者正交不冲突，可同时使用（一个横切模型层，一个竖切矩阵），核心目的是解决单个GPU显存不足的问题。

### （四）三者协同特性
- 兼容性：expert parallel、Tensor parallel、pipeline parallel可组合使用，例如可对单个expert的矩阵进一步拆分（Tensor parallel），同时按层分布任务（pipeline parallel）。
- 通信损耗：单台机器内GPU间通信损耗较小，多台机器通过光纤连接时损耗会显著增大；建模时需关注`tokens数量×hidden size×2÷EP进程数×网络带宽`的通信开销计算公式，但多数文献中暂不重点考虑通信开销，建议在后续优化阶段补充评估。

## 四、代码实现与实验问题
### （一）核心代码相关
- EP（expert parallel）通过all to all通信实现，TP（Tensor parallel）在Linear层外套Wrapper实现，二者正交可独立建模。
- 需重点核查：EP代码位置、routing调用逻辑、expert selection的负载均衡实现（参考`export placement`相关代码）。

### （二）expert replication（专家副本）
- VLM版本现状：0.1 3.0版本未实现expert副本激活和负载感知路由，但支持部分动态显存管理功能；最新0.1 4.0版本新增async scheduling和新模型架构，需进一步适配。
- 实现路径：可通过修改VLM代码补充副本激活机制，核心思路为激活hard expert，基于负载感知路由选择通信消耗最小的expert执行任务（参考efficient moe相关文献）。

### （三）实验参考
- 相关文献：Metro（Moe serving方向）提出通过TOKEN路由至已激活expert，减少expert启动耗时；港中文相关survey探讨expert duplicate策略。
- 对比分析：部分现有工作（如千问、Deepseek）基于VLM实现，但存在简化底层逻辑的问题，需结合代码深入验证。

## 五、后续工作安排
### （一）线下交流
- 时间：明天下午2点半（餐后开始），预计交流3小时，5点半前结束（方便返程）。
- 地点：学校（具体位置另行确认）。
- 核心目标：高效对齐VLM通信流程、并行机制细节，梳理代码实现关键点。

### （二）资料整理与任务分配
| 任务内容 | 负责人 | 截止时间 |
|----------|--------|----------|
| 整理会议讨论内容（含EP通信开销、expert replication等）至Markdown文档 | 说话人A | 会议结束后1天内 |
| 准备VLM文件夹（删繁就简） | 说话人B | 明天线下交流前 |
| 阅读`VVV 1 call course share schedule`类代码，分析`scheduler output`生成逻辑 | 说话人A | 下次交流前 |
| 核查GPU model Runner与GPU worker的架构类型（物理/软件） | 共同完成 | 明天线下交流期间 |
| 分享OpenAI重构AI Infra的Podcast链接 | 说话人A | 会议结束后即时 |

## 六、其他补充
- 基金与股市：说话人A提及当日基金亏损220元（电池板块拖累），美股同步下跌，全年基金+股票亏损2000出头，该部分为闲聊内容，不涉及工作推进。