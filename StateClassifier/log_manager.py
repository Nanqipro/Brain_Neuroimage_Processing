#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志管理工具

该模块提供日志文件的管理和查看功能，包括清理旧日志、查看最新日志等。

作者: Clade 4
日期: 2025年5月23日
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import glob

from config import config


def list_log_files():
    """
    列出所有日志文件
    """
    print("=" * 60)
    print("当前日志文件列表")
    print("=" * 60)
    
    if not config.LOGS_DIR.exists():
        print("日志目录不存在")
        return
    
    log_files = list(config.LOGS_DIR.glob("*.log"))
    
    if not log_files:
        print("没有找到日志文件")
        return
    
    for log_file in sorted(log_files):
        file_stat = log_file.stat()
        size = file_stat.st_size
        modified = datetime.fromtimestamp(file_stat.st_mtime)
        
        # 格式化文件大小
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        
        print(f"📄 {log_file.name}")
        print(f"   大小: {size_str}")
        print(f"   修改时间: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        print()


def show_latest_logs(log_file: str = None, lines: int = 50):
    """
    显示最新的日志内容
    
    Parameters
    ----------
    log_file : str, optional
        指定的日志文件名，如果为None则显示主日志文件
    lines : int, default=50
        显示的行数
    """
    if log_file is None:
        target_file = config.MAIN_LOG_FILE
    else:
        target_file = config.LOGS_DIR / log_file
    
    if not target_file.exists():
        print(f"日志文件不存在: {target_file}")
        return
    
    print("=" * 60)
    print(f"最新日志内容: {target_file.name}")
    print(f"显示最后 {lines} 行")
    print("=" * 60)
    
    try:
        with open(target_file, 'r', encoding='utf-8') as f:
            lines_list = f.readlines()
            for line in lines_list[-lines:]:
                print(line.rstrip())
    except UnicodeDecodeError:
        # 如果UTF-8解码失败，尝试其他编码
        try:
            with open(target_file, 'r', encoding='gbk') as f:
                lines_list = f.readlines()
                for line in lines_list[-lines:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"读取日志文件失败: {e}")


def clear_old_logs(days: int = 30):
    """
    清理超过指定天数的旧日志文件
    
    Parameters
    ----------
    days : int, default=30
        保留天数，超过这个天数的日志文件将被删除
    """
    if not config.LOGS_DIR.exists():
        print("日志目录不存在")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    log_files = list(config.LOGS_DIR.glob("*.log"))
    
    deleted_count = 0
    for log_file in log_files:
        file_stat = log_file.stat()
        modified = datetime.fromtimestamp(file_stat.st_mtime)
        
        if modified < cutoff_date:
            print(f"删除旧日志文件: {log_file.name} (修改时间: {modified.strftime('%Y-%m-%d %H:%M:%S')})")
            log_file.unlink()
            deleted_count += 1
    
    if deleted_count == 0:
        print(f"没有找到超过 {days} 天的日志文件")
    else:
        print(f"已删除 {deleted_count} 个旧日志文件")


def backup_logs():
    """
    备份当前日志文件
    """
    if not config.LOGS_DIR.exists():
        print("日志目录不存在")
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = config.LOGS_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    log_files = list(config.LOGS_DIR.glob("*.log"))
    
    if not log_files:
        print("没有找到需要备份的日志文件")
        return
    
    copied_count = 0
    for log_file in log_files:
        backup_file = backup_dir / log_file.name
        backup_file.write_bytes(log_file.read_bytes())
        copied_count += 1
    
    print(f"已备份 {copied_count} 个日志文件到: {backup_dir}")


def create_logs_directory():
    """
    创建日志目录
    """
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 日志目录已创建: {config.LOGS_DIR}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="日志管理工具")
    parser.add_argument("--list", action="store_true", help="列出所有日志文件")
    parser.add_argument("--show", metavar="LOG_FILE", help="显示指定日志文件的最新内容")
    parser.add_argument("--lines", type=int, default=50, help="显示的行数 (默认: 50)")
    parser.add_argument("--clear", type=int, metavar="DAYS", help="清理超过指定天数的旧日志")
    parser.add_argument("--backup", action="store_true", help="备份当前日志文件")
    parser.add_argument("--create-dir", action="store_true", help="创建日志目录")
    
    args = parser.parse_args()
    
    if args.create_dir:
        create_logs_directory()
    elif args.list:
        list_log_files()
    elif args.show is not None:
        show_latest_logs(args.show if args.show else None, args.lines)
    elif args.clear is not None:
        clear_old_logs(args.clear)
    elif args.backup:
        backup_logs()
    else:
        # 默认显示主日志文件
        list_log_files()
        print()
        show_latest_logs(lines=20)


if __name__ == "__main__":
    main() 