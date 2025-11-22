import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-app.js";
import { getAuth, signInWithEmailAndPassword } from "https://www.gstatic.com/firebasejs/11.6.0/firebase-auth.js";

const firebaseConfig = {
    apiKey: "AIzaSyCT1WH_i7HTXDdOVBIDLlWkr6Yl86P7PmU",
    authDomain: "smartlide.firebaseapp.com",
    projectId: "smartlide",
    storageBucket: "smartlide.appspot.com",
    messagingSenderId: "697955025928",
    appId: "1:697955025928:web:e04385bc766f5bf5667dbd"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("login-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        signInWithEmailAndPassword(auth, email, password)
            .then((userCredential) => {
                console.log("登入成功");
                window.location.href = "/home"; // 確保你的 Django 有設置這個路由
            })
            .catch((error) => {
                console.error("登入錯誤：", error.message);
                document.getElementById("message").innerText = "帳號或密碼有誤";
            });
    });
});
