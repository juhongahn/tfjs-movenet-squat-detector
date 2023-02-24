import { useEffect, useRef } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu'; // cpu 백엔드 추가

// Register one of the TF.js backends.


export default function Home() {
  const videoRef = useRef();

  useEffect(() => {
    async function runSquatDetector() {

      if (tf.getBackend() === 'webgl' || tf.getBackend() === 'webgpu') {
        await tf.setBackend('webgl');
      } else {
        await tf.setBackend('cpu');
      }
      const model = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;


      const SQUAT_REPS = 5; // 목표 스쿼트 반복 횟수
      let squatCount = 0; // 현재까지 인식된 스쿼트 반복 횟수
      let squatDetected = false; // 스쿼트 동작이 인식되었는지 여부

      async function detectSquat() {


        const image = tf.browser.fromPixels(videoRef.current);
        const pose = await model.estimatePoses(image, { flipHorizontal: false });

        if (pose.length > 0) {
          const leftHip = pose[0].keypoints.find((k) => k.name === 'leftHip');
          const rightHip = pose[0].keypoints.find((k) => k.name === 'rightHip');
          const leftKnee = pose[0].keypoints.find((k) => k.name === 'leftKnee');
          const rightKnee = pose[0].keypoints.find((k) => k.name === 'rightKnee');

          if (
            leftHip && rightHip && leftKnee && rightKnee &&
            leftHip.score > 0.5 && rightHip.score > 0.5 && leftKnee.score > 0.5 && rightKnee.score > 0.5
          ) {
            const hipWidth = rightHip.x - leftHip.x;
            const kneeDistance = Math.abs(rightKnee.y - leftKnee.y);

            if (kneeDistance < 0.8 * hipWidth) {
              if (!squatDetected) {
                squatCount++;
                console.log('squat');
                squatDetected = true;
              }
            } else {
              squatDetected = false;
            }
          } else {
            squatDetected = false;
          }

          if (squatCount >= SQUAT_REPS) {
            console.log(`🎉 축하합니다! 목표 스쿼트 ${SQUAT_REPS}회를 달성했습니다.`);
            return;
          }
        }

        requestAnimationFrame(detectSquat);
      }

      videoRef.current.addEventListener('loadedmetadata', () => {
        detectSquat();
      });
    }

    runSquatDetector();
  }, []);

  return (
    <>
      <video ref={videoRef} autoPlay width={600} height={450} />
    </>
  );
}
