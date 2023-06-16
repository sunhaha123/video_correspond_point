import librosa
import numpy as np
import matplotlib.pyplot as plt

# 加载音频数据
y, sr = librosa.load('F:/1code/video_corr_point/example.mp3', sr=None)

# 计算音频的强度谱
o_env = librosa.onset.onset_strength(y=y, sr=sr)

# 获取与强度谱相对应的时间点
times = librosa.times_like(o_env, sr=sr)

# 检测音频中的节奏起始点
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

# 计算音频的幅度谱
D = np.abs(librosa.stft(y))

# 创建图形窗口
fig, ax = plt.subplots(nrows=2, sharex=True)

# 显示音频的功率谱图
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()

# 绘制音频的强度谱和节奏起始点
ax[1].plot(times, o_env, label='Onset strength')
ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
             linestyle='--', label='Onsets')
ax[1].legend()

# 显示图形
plt.show()
