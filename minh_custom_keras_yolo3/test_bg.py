import cv2

bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
camera = cv2.VideoCapture("/home/minhdq99hp/Desktop/iot-camera/cuahang1.mp4")

ret, frame = camera.read()
while ret:
	ret, frame = camera.read()
	fg = bs.apply(frame)
	if fg is not None:
		th = cv2.threshold(fg.copy(), 244, 255, cv2.THRESH_BINARY)[1]
		dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)

		image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for c in contours:
			if cv2.contourArea(c) > 1600:
				(x,y,w,h) = cv2.boundingRect(c)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2)

		cv2.imshow("mog", fg)
		cv2.imshow("thresh", th)
		cv2.imshow("detection", frame)

	if cv2.waitKey(30) & 0xFF == 27:
		break

	camera.release()
	cv2.destroyAllWindows()