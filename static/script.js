document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const resultContent = document.getElementById('result-content');
    const progressBarContainer = document.getElementById('progress-bar-container');
    const accuracyDisplay = document.getElementById('accuracy');
    const predictionDisplay = document.getElementById('prediction');
    const uploadedImg = document.getElementById('uploaded-img');
    const processedImg = document.getElementById('processed-img');
    const downloadBtn = document.getElementById('download-report-btn');
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const disclaimerPopup = document.getElementById('disclaimer-popup');
    const closePopupBtn = document.getElementById('close-popup-btn');

    let latestPrediction = null;

    // Tampilkan pop-up disclaimer hanya sekali per sesi
    if (!sessionStorage.getItem('disclaimerShown')) {
        disclaimerPopup.classList.add('show');
        sessionStorage.setItem('disclaimerShown', 'true');
    }

    // Tutup pop-up saat tombol "Understood" diklik
    closePopupBtn.addEventListener('click', () => {
        disclaimerPopup.classList.remove('show');
    });

    // Drag and drop functionality
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        imageInput.files = e.dataTransfer.files;
        if (imageInput.files[0]) {
            uploadedImg.src = URL.createObjectURL(imageInput.files[0]);
            uploadedImg.style.display = 'block';
        }
    });

    dropZone.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files[0]) {
            uploadedImg.src = URL.createObjectURL(imageInput.files[0]);
            uploadedImg.style.display = 'block';
        }
    });

    // Dark mode toggle
    darkModeToggle.addEventListener('change', () => {
        document.body.classList.toggle('dark-mode');
    });

    // Predict button click
    predictBtn.addEventListener('click', async (e) => {
        e.preventDefault();

        if (!imageInput.files.length) {
            alert('Please upload an image.');
            return;
        }

        const file = imageInput.files[0];
        const modelChoice = document.getElementById('model-choice').value;

        // Reset and show loader
        resultSection.classList.remove('show');
        resultContent.style.display = 'none';
        resultSection.classList.add('loading');
        progressBarContainer.style.display = 'flex';
        resultSection.style.opacity = '1';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', modelChoice);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Prediction failed');
            }

            console.log("Frontend received:", data);
            latestPrediction = data;

            await new Promise(resolve => setTimeout(resolve, 2000));

            resultSection.classList.remove('loading');
            progressBarContainer.style.display = 'none';
            resultSection.classList.add('show');
            resultContent.style.display = 'block';

            const accuracy = data.accuracy !== undefined ? data.accuracy.toFixed(2) : 'N/A';
            accuracyDisplay.textContent = `${accuracy}%`;
            predictionDisplay.innerHTML = `<span class="${data.prediction === 'Tumor Detected' ? 'tumor' : 'no-tumor'}">${data.prediction}</span>`;
            processedImg.src = URL.createObjectURL(file);
            processedImg.style.display = 'block';

        } catch (error) {
            resultSection.classList.remove('loading');
            progressBarContainer.style.display = 'none';
            alert(`Error: ${error.message}`);
            resultContent.style.display = 'none';
        }
    });

    // Download report
    downloadBtn.addEventListener('click', () => {
        if (!latestPrediction) {
            alert('Please run a prediction first.');
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        doc.setFontSize(18);
        doc.setTextColor(27, 94, 32);
        doc.text("NeuroTori - Brain Tumor Detection Report", 10, 20);

        doc.setFontSize(12);
        doc.setTextColor(0, 0, 0);
        doc.text(`Date: ${new Date().toLocaleString()}`, 10, 30);
        doc.text(`Prediction: ${latestPrediction.prediction}`, 10, 40);
        doc.text(`Accuracy: ${latestPrediction.accuracy.toFixed(2)}%`, 10, 50);

        doc.addImage(uploadedImg.src, 'JPEG', 10, 60, 50, 50, undefined, 'FAST');
        doc.addImage(processedImg.src, 'JPEG', 70, 60, 50, 50, undefined, 'FAST');

        doc.setFontSize(10);
        doc.setTextColor(100, 100, 100);
        doc.text('Generated by NeuroTori - Powered by HitoVG-16T', 10, 290);

        doc.save('NeuroTori_Report.pdf');
    });
});