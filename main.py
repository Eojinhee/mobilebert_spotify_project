import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("✅ MobileBERT Spotify 프로젝트 실행")

print("MobileBERT 학습시작")
with open("mobilebert_project.py", encoding="utf-8") as f:
    exec(f.read())


print("모든 작업이 완료되었습니다.")
