"""
Entry point for running dissector as a module: python -m dissector
"""
import os
import uvicorn
from .app import app


def main():
    """主函数"""
    print("Starting Dissector service...")
    print("Models will be loaded on startup...")
    
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')


if __name__ == '__main__':
    main()

