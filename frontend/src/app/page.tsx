"use client";

import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

export default function Home() {
  const [showCam, setShowCam] = useState(false);
  const webcamRef = useRef<Webcam>(null); // to interact with the webcam instance
  const [screenshot, setScreenshot] = useState<string | null>(null);

  useEffect(() => {
    if (showCam) {
      const timer = setTimeout(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          setScreenshot(imageSrc || null);
          console.log("Screen shot taken!");
        }
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [showCam]);

  useEffect(()=> {
    if(screenshot){
      setShowCam(false);

    }
  }, [screenshot]);

  return (
    // needs height for centering
    <div className="flex flex-col justify-center items-center h-screen gap-4">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-sky-400 to-indigo-500 text-transparent bg-clip-text p-10 rounded-xl">
        Facial authentication website
      </h1>
      {!showCam && !screenshot && (
        <button
          className="border rounded-2xl p-3 hover:bg-sky-700"
          onClick={() => setShowCam(true)}
        >
          Login
        </button>
      )}

      {/* show cam sections */}
      {showCam && (
        <div className="flex flex-col justify-center items-center gap-3">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: "user" }}
            className="rounded-xl shadow-lg"
          />

          <button
            className="bg-white w-24 text-black rounded-xl h-12 hover:bg-yellow-500"
            onClick={() => {
              setShowCam(false);
              setScreenshot(null);
            }}
          >
            Cancle{" "}
          </button>
        </div>
      )}

      {/* picture screenshot */}
      {screenshot && (
        <div>
        <img
          src={screenshot}
          alt="Screenshot"
          className="rounded-xl max-w-xs"
        ></img>
        </div>
      )}
    </div>
  );
}
