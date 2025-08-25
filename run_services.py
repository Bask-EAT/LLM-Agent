#!/usr/bin/env python3
"""
마이크로서비스 실행 스크립트
"""

import subprocess
import time
import sys
import os

def start_service(service_name, service_path, port):
    """서비스 시작"""
    print(f"🚀 {service_name} 시작 중... (포트: {port})")
    try:
        # 서비스 디렉토리로 이동하여 실행
        process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=service_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"✅ {service_name} 시작됨 (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ {service_name} 시작 실패: {e}")
        return None

def main():
    """모든 서비스 시작"""
    print("🏗️ 마이크로서비스 시작")
    print("=" * 50)
    
    # 서비스 목록
    services = [
        {
            "name": "Intent LLM Service",
            "path": "intent_service",
            "port": 8001
        },
        {
            "name": "TextAgent Service", 
            "path": "text_service",
            "port": 8002
        },
        {
            "name": "VideoAgent Service",
            "path": "video_service",
            "port": 8003
        }
    ]
    
    processes = []
    
    try:
        # 각 서비스 시작
        for service in services:
            process = start_service(service["name"], service["path"], service["port"])
            if process:
                processes.append(process)
            time.sleep(2)  # 서비스 시작 간격
        
        print("\n" + "=" * 50)
        print("🎉 모든 서비스가 시작되었습니다!")
        print("\n📋 서비스 정보:")
        print("• Intent LLM Service: http://localhost:8001")
        print("• TextAgent Service: http://localhost:8002")
        print("• VideoAgent Service: http://localhost:8003")
        print("\n💡 메인 앱 실행: cd main_app && python app.py")
        print("💡 서비스 중지: Ctrl+C")
        print("=" * 50)
        
        # 무한 대기
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 서비스 중지 중...")
        for process in processes:
            if process:
                process.terminate()
                print(f"✅ 프로세스 종료됨 (PID: {process.pid})")
        print("👋 모든 서비스가 종료되었습니다.")

if __name__ == "__main__":
    main() 