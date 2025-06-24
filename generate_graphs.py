import matplotlib.pyplot as plt

train_loss = [0.65, 0.48, 0.42]
plt.figure()
plt.plot(range(1, 4), train_loss, marker='o', color='blue')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.close()

val_acc = [0.72, 0.81, 0.84]
plt.figure()
plt.plot(range(1, 4), val_acc, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("validation_accuracy.png")
plt.close()

months = ["2022-01", "2022-02", "2022-03", "2022-04"]
counts = [120, 340, 560, 450]
plt.figure()
plt.plot(months, counts, marker='o', color='purple')
plt.title("월별 리뷰 수 변화 추이")
plt.xlabel("월")
plt.ylabel("리뷰 수")
plt.grid(True)
plt.savefig("monthly_review_count.png")
plt.close()

print("그래프 3장이 루트 디렉토리에 저장되었습니다.")
