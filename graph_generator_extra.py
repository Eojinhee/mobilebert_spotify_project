import matplotlib.pyplot as plt
import numpy as np

labels = ['긍정', '부정']
counts = [45000, 16000]
colors = ['#4CAF50', '#F44336']

plt.figure()
plt.bar(labels, counts, color=colors)
plt.title('긍정 / 부정 리뷰 분포')
plt.ylabel('리뷰 수')
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.close()

np.random.seed(0)
review_lengths = np.random.normal(loc=100, scale=30, size=2000)

plt.figure()
plt.hist(review_lengths, bins=30, color='#2196F3', edgecolor='black')
plt.title('리뷰 길이 분포')
plt.xlabel('리뷰 길이 (글자 수)')
plt.ylabel('리뷰 수')
plt.tight_layout()
plt.savefig("review_length_dist.png")
plt.close()

print("그래프 2종 생성 완료: label_distribution.png, review_length_dist.png")
