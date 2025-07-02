#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from decimal import Decimal, getcontext

def compute_power(r_str: str, n: int) -> str:
    # 1) 计算 R 的有效数字长度，用来估算结果需要的精度
    digits = r_str.replace('.', '')
    total_significant = len(digits) * n
    # 多留一点余量，防止中途运算截断
    getcontext().prec = total_significant + 5

    # 2) 精确幂运算
    dec = Decimal(r_str) ** n

    # 3) 转为定点字符串，去掉无用的零
    s = format(dec, 'f').strip('0').rstrip('.')
    # 特殊情况：全部被去光了，变成空串，就输出 "0"
    return s or "0"

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: 
            continue
        r_str, n_str = line.split()
        print(compute_power(r_str, int(n_str)))

if __name__ == "__main__":
    main()
