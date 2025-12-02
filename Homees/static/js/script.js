// ----------------------
// Flash Messages (keep your original)
document.addEventListener('DOMContentLoaded', () => {
    const flashes = document.querySelectorAll('.flashed-message');
    flashes.forEach(flash => {
        const message = flash.dataset.message;
        if (message) {
            alert(message);
        }
        flash.remove(); 
    });
});

// ----------------------
// OTP Timer (keep your original)
const timerEl = document.getElementById('timer');
const resendBtn = document.getElementById('resendBtn');

if (timerEl && resendBtn) {
    let timeLeft = parseInt(timerEl.dataset.time);

    const countdown = setInterval(() => {
        if (timeLeft <= 0) {
            clearInterval(countdown);
            timerEl.textContent = "Expired";
            resendBtn.style.display = "block";
        } else {
            timerEl.textContent = timeLeft;
            timeLeft--;
        }
    }, 1000);

    resendBtn.addEventListener('click', () => {
        const url = resendBtn.dataset.url;
        fetch(url)
            .then(response => response.json())
            .then(data => {
                alert(data.message);
                timeLeft = parseInt(timerEl.dataset.time);
                timerEl.textContent = timeLeft;
                resendBtn.style.display = "none";
                const innerCountdown = setInterval(() => {
                    if (timeLeft <= 0) {
                        clearInterval(innerCountdown);
                        timerEl.textContent = "Expired";
                        resendBtn.style.display = "block";
                    } else {
                        timerEl.textContent = timeLeft;
                        timeLeft--;
                    }
                }, 1000);
            });
    });
}

// ----------------------
// Dashboard Upload -> Backend API
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const fileNameSpan = document.getElementById("fileName");
const resultMessages = document.getElementById("resultMessages");

// Show selected file name
fileInput.addEventListener("change", () => {
    fileNameSpan.textContent = fileInput.files[0]?.name || "No file chosen";
});

// Handle form submission
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault(); // prevent default form submission

    if (!fileInput.files.length) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultMessages.innerHTML = "<li class='msg'>Uploading...</li>";

    try {
        // Ensure the request goes to the correct backend port
        const response = await fetch("http://127.0.0.1:5001/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        resultMessages.innerHTML = "";

        if (data.error) {
            resultMessages.innerHTML = `<li class="msg danger">${data.error}</li>`;
        } else {
            resultMessages.innerHTML = `<li class="msg success">✅ Bone X-ray detected! (${(data.confidence*100).toFixed(2)}%)</li>`;
        }
    } catch (err) {
        console.error(err);
        resultMessages.innerHTML = `<li class="msg danger">⚠️ Error uploading file</li>`;
    }
});
