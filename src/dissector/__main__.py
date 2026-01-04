"""
Entry point for running dissector as a module: python -m dissector
"""
import os
import uvicorn


def main():
    print("Starting Dissector service...")
    print("Models will be loaded on startup...")
    
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    
    if debug:
        uvicorn.run("dissector.app:app", host='0.0.0.0', port=port, log_level='info', reload=True)
    else:
        from .app import app
        uvicorn.run(app, host='0.0.0.0', port=port, log_level='info', reload=False)


if __name__ == '__main__':
    main()

