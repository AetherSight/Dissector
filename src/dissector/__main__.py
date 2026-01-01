"""
Entry point for running dissector as a module: python -m dissector
"""
import os
import uvicorn


def main():
    """主函数"""
    print("Starting Dissector service...")
    print("Models will be loaded on startup...")
    
    port = int(os.getenv('PORT', 8000))
    reload = os.getenv('RELOAD', 'true').lower() == 'true'
    
    if reload:
        # 使用导入字符串以启用自动重载
        uvicorn.run("dissector.app:app", host='0.0.0.0', port=port, log_level='info', reload=reload)
    else:
        # 不使用reload时可以直接导入app对象
        from .app import app
        uvicorn.run(app, host='0.0.0.0', port=port, log_level='info', reload=reload)


if __name__ == '__main__':
    main()

