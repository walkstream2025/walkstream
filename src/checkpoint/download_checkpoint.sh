#!/bin/bash

# 检查当前所处路径
if [ "${PWD##*/}" != "checkpoint" ]; then
    echo "应切换到 checkpoint 文件夹下解压"
    exit 1
fi


# 下载数据集
echo " checkpoint下载 .."
coscmd download -r leonchadli/checkpoint/ ./checkpoint/


echo " checkpoint下载完毕"