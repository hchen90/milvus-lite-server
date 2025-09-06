#!/usr/bin/env python3
"""
Milvus Lite Server 测试运行器
自动运行 app/tests 文件夹下的所有测试用例
"""

import sys
import os
import unittest
import time
import logging
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """测试运行器类"""
    
    def __init__(self):
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.tests_dir = os.path.join(self.app_dir, 'tests')
        self.project_root = os.path.dirname(self.app_dir)
        
        # 添加路径到Python搜索路径
        self._setup_python_path()
    
    def _setup_python_path(self):
        """设置Python路径"""
        paths_to_add = [self.app_dir, self.project_root]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
    
    def discover_test_files(self) -> List[str]:
        """发现所有测试文件"""
        test_files = []
        
        if not os.path.exists(self.tests_dir):
            print(f"测试目录不存在: {self.tests_dir}")
            return test_files
        
        for file_path in Path(self.tests_dir).rglob("test_*.py"):
            test_files.append(str(file_path))
        
        return sorted(test_files)
    
    def print_banner(self):
        """打印横幅信息"""
        print("=" * 60)
        print("  Milvus Lite Server - 测试运行器")
        print("=" * 60)
        print(f"项目根目录: {self.project_root}")
        print(f"应用目录: {self.app_dir}")
        print(f"测试目录: {self.tests_dir}")
        print("=" * 60)
    
    def run_all_tests(self, verbosity: int = 2) -> bool:
        """运行所有测试"""
        self.print_banner()
        
        # 发现测试文件
        test_files = self.discover_test_files()
        
        if not test_files:
            print("❌ 没有找到任何测试文件 (test_*.py)")
            print("请确保测试文件名以 'test_' 开头")
            return False
        
        print(f"📁 发现 {len(test_files)} 个测试文件:")
        for test_file in test_files:
            relative_path = os.path.relpath(test_file, self.project_root)
            print(f"   ✓ {relative_path}")
        print()
        
        # 使用unittest的测试发现功能
        start_time = time.time()
        
        try:
            # 创建测试套件
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            # 从tests目录加载所有测试
            discovered_tests = loader.discover(
                start_dir=self.tests_dir,
                pattern='test_*.py',
                top_level_dir=self.app_dir
            )
            
            suite.addTest(discovered_tests)
            
            # 运行测试
            runner = unittest.TextTestRunner(
                verbosity=verbosity,
                stream=sys.stdout,
                descriptions=True,
                failfast=False
            )
            
            print("🚀 开始运行测试...")
            print("-" * 60)
            
            result = runner.run(suite)
            
            # 显示测试结果统计
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print("📊 测试结果统计")
            print("=" * 60)
            print(f"运行时间: {duration:.2f} 秒")
            print(f"总测试数: {result.testsRun}")
            print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
            print(f"失败: {len(result.failures)}")
            print(f"错误: {len(result.errors)}")
            print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
            
            # 显示失败和错误的详细信息
            if result.failures:
                print("\n❌ 失败的测试:")
                for test, traceback in result.failures:
                    print(f"   • {test}")
            
            if result.errors:
                print("\n💥 错误的测试:")
                for test, traceback in result.errors:
                    print(f"   • {test}")
            
            # 判断是否所有测试都通过
            success = len(result.failures) == 0 and len(result.errors) == 0
            
            if success:
                print("\n🎉 所有测试都通过了!")
            else:
                print("\n😞 有测试失败或出错")
            
            print("=" * 60)
            
            return success
            
        except Exception as e:
            print(f"❌ 运行测试时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_specific_test(self, test_name: str, verbosity: int = 2) -> bool:
        """运行特定的测试"""
        self.print_banner()
        
        try:
            # 尝试导入并运行特定测试
            suite = unittest.TestSuite()
            loader = unittest.TestLoader()
            
            # 如果test_name包含模块路径
            if '.' in test_name:
                suite.addTest(loader.loadTestsFromName(test_name))
            else:
                # 在tests目录中查找测试
                test_file = os.path.join(self.tests_dir, f"{test_name}.py")
                if os.path.exists(test_file):
                    module_name = f"tests.{test_name}"
                    suite.addTest(loader.loadTestsFromName(module_name))
                else:
                    print(f"❌ 找不到测试文件: {test_file}")
                    return False
            
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            
            return len(result.failures) == 0 and len(result.errors) == 0
            
        except Exception as e:
            print(f"❌ 运行测试 '{test_name}' 时出错: {e}")
            return False
    
    def list_tests(self):
        """列出所有可用的测试"""
        self.print_banner()
        
        test_files = self.discover_test_files()
        
        if not test_files:
            print("❌ 没有找到任何测试文件")
            return
        
        print("📋 可用的测试文件:")
        for test_file in test_files:
            relative_path = os.path.relpath(test_file, self.project_root)
            module_name = os.path.splitext(os.path.basename(test_file))[0]
            print(f"   • {relative_path} (模块名: {module_name})")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Milvus Lite Server 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python app/test.py                    # 运行所有测试
  python app/test.py --list             # 列出所有测试
  python app/test.py --test test_config # 运行特定测试
  python app/test.py --verbose 1        # 设置详细程度
        """
    )
    
    parser.add_argument(
        '--test', '-t',
        type=str,
        help='运行特定的测试 (例如: test_config)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用的测试'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='设置详细程度 (0=静默, 1=正常, 2=详细)'
    )
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose == 2 else logging.WARNING if args.verbose == 0 else logging.ERROR
    )
    
    runner = TestRunner()
    
    if args.list:
        runner.list_tests()
        return
    
    if args.test:
        success = runner.run_specific_test(args.test, args.verbose)
    else:
        success = runner.run_all_tests(args.verbose)
    
    # 退出码: 0表示成功，1表示失败
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
