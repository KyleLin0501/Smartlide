// 引入 Firebase 模組
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-auth.js";

// Firebase 設定
const firebaseConfig = {
    apiKey: "AIzaSyCT1WH_i7HTXDdOVBIDLlWkr6Yl86P7PmU",
    authDomain: "smartlide.firebaseapp.com",
    projectId: "smartlide",
    storageBucket: "smartlide.appspot.com",
    messagingSenderId: "697955025928",
    appId: "1:697955025928:web:e04385bc766f5bf5667dbd",
    measurementId: "G-3FW4861Z8B"
};

// 初始化 Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// 註冊表單提交處理
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("register-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        createUserWithEmailAndPassword(auth, email, password)
            .then((userCredential) => {
                const user = userCredential.user;
                document.getElementById("message").innerText = "✅ 註冊成功！";
                console.log("註冊成功", user);
            })
            .catch((error) => {
                document.getElementById("message").innerText = "❌ 錯誤：" + error.message;
                console.error("註冊錯誤", error);
            });
    });
});
