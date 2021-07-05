import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
########### 카메라 대신 youtube영상으로 대체 ############
import pafy
url = 'https://www.youtube.com/watch?v=u_Q7Dkl7AIk'
video = pafy.new(url)
print('title = ', video.title)
print('video.rating = ', video.rating)
print('video.duration = ', video.duration)
 
best = video.getbest(preftype='mp4')     # 'webm','3gp'
print('best.resolution', best.resolution)
 
#cap=cv2.VideoCapture(best.url)
#########################################################
 
class Video:
    # 첫 프레임을 가져오기(맛보기용, 정보 확인용)
    def __init__(self, device=0):
        self.cap=cv2.VideoCapture(best.url)
        #self.cap = cv2.VideoCapture(device)	# 카메라로 영상 가져올 때 사용
        self.retval, self.frame = self.cap.read()
        self.im = plt.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        print('start capture ...')
       
    def updateFrame(self, k):
        self.retval, self.frame = self.cap.read()
        self.im.set_array(cv2.cvtColor(camera.frame, cv2.COLOR_BGR2RGB))
        # return self.im,
 
    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        print('finish capture.')
 
# 프로그램 시작 
fig = plt.figure()	# 플롯(frame, 그래프)이 업데이트 될 객체
fig.canvas.set_window_title('Video Capture')	# fig의 제목(타이틀)
plt.axis("off")	# 눈금 표시 안하기
 
camera = Video()	# Video 클래스 생성
ani = animation.FuncAnimation(fig, camera.updateFrame, interval=50)
plt.show()
camera.close()
