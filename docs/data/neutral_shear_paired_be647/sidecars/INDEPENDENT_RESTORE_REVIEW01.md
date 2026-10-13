# 选定原文件的独立字节恢复

结论：**PASS_INDEPENDENT_ORIGINAL_BYTE_RESTORE**。Lady / arithmetic_static_fresh 独立于 packer/portable reader 作者，完整读取准备规格与代码后，在 root 的一次授权下调用便携 reader 恰好一次，没有重试，也没有再次执行 selection、pack 或旧恢复。

真实性由工作进程和父进程共同确认：worker 的字节检查通过；父进程实际 exit 0、stop reason 为 null，墙钟 `68.40982751099999` s，341 次约 0.2 s 的采样中，子进程组峰值 RSS 为 `46821376` bytes。约束为一 CPU、180 s、4 GiB，另继承每进程 RLIMIT_AS；进程组 RSS 是采样而非硬 cgroup，不能排除采样之间更短的峰值。

独立清单精确匹配 841 个选定原文件。33 个分卷逐一哈希及拼接整包哈希正确，ZIP 成员无越界、重复、大小写碰撞、符号链接或加密/压缩替换；恢复目录只包含该精确文件集及 reader 自身的 `RESTORE_RECEIPT.json`。所有 841 个文件的原 SHA（恢复前）、原 SHA（恢复后）、恢复 SHA 与锁定 SHA 完全一致，原 size/mtime 也未改变。9,461 项元数据/原字节检查均通过。

覆盖包含完整 installed 133 个 Python 文件、五组原生/初态/audit 原文件、205 个原始逐输出归档、失败初始化前身、23 个解析桥接原证据，以及旧 source70 的五 verdict、plan、definitions 和六 core，共 13 个元数据/核心文件。旧 source70 数值数组不在所选范围。包不是整个工作目录：晚写报告、frame/报告元数据、selection/pack/restore 自身运行凭据及缓存按公开选择规则排除，不能继承快照校验和。

| 凭据 | SHA256 |
| --- | --- |
| [独立恢复结论](INDEPENDENT_RESTORE_REVIEW01.json) | `c67f91869e3fd5cac2fc047b7aa4da857c6563610b12fc3c11d7a4a62c5e2205` |
| [逐文件详细结果](INDEPENDENT_RESTORE_VERDICT01.json) | `10a09a0a64849524de545b42c8646264cbf6e95685c3853b2d3dee90404d1ffb` |
| [父进程资源](independent-restore01-resources.json) | `de6598c327c7c0a16842c2539c70c2dcad2ddf8a15916a4c8c95c23d0baf8cc8` |
| [一次性声明](INDEPENDENT_RESTORE_DECLARATION01.json) | `f4baded42e20e2659297599a24456e1973a87f2a00477b0c390bc099db2dc79a` |
| 原 manifest | `476ddf2848d6a6ab341d6b7f40223f287a3ad19f3a1ba4533c9fe96bcca48796` |
| 原 selection | `57c8517de678ae4921490af23fa76464f0165a574a83127d601b38065181b603` |
| 原整包 ZIP | `79747029cdf37d73f79f0263f6faa5b38a9e586fa7c866140656e4f7cbcab8ab` |
| reader 原字节 | `37c6dac738db527495f389708dc906de5a3d47a71d653fe5b82c87f384e9b4ac` |

新目录 `publication-preparation/independent-restored01` 与同级 `independent-restored01-joined.zip` 完整保留。读取的是原始文件字节和外层 ZIP 元数据；未解码 NPZ、导入 NumPy/模型/科学 oracle、重算指标或改变原证据。这一结论只证明所选字节可以正确恢复，不新增任何数值、PDE、轨迹、holdout 或排名资格。原中文 draft04 保持冻结，恢复完成说明是快照之后的独立附加材料。
