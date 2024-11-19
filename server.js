const express = require('express');
const multer = require('multer');
const admin = require('firebase-admin');
const { Storage } = require('@google-cloud/storage');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const upload = multer({ limits: { fileSize: 1000000 } }); // Maksimal 1MB

// Inisialisasi Firebase Admin
const serviceAccount = require('./path/to/serviceAccountKey.json'); // Ganti dengan path ke file service account Anda
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});
const db = admin.firestore();

// Load model dari Cloud Storage
let model;
async function loadModel() {
    model = await tf.loadGraphModel('gs://<bucket-name>/model.json'); // Ganti <bucket-name> dengan nama bucket Anda
}
loadModel();

app.post('/predict', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi'
        });
    }

    try {
        // Proses gambar dan prediksi
        const imageTensor = tf.node.decodeImage(req.file.buffer);
        const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]).expandDims(0).toFloat().div(255);
        const predictions = await model.predict(resizedImage).data();

        const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        const id = uuidv4(); // Menghasilkan ID unik

        // Simpan hasil ke Firestore
        await db.collection('predictions').doc(id).set({
            id: id,
            result: result,
            suggestion: suggestion,
            createdAt: new Date().toISOString()
        });

        res.status(200).json({
            status: 'success',
            message: 'Model is predicted successfully',
            data: {
                id: id,
                result: result,
                suggestion: suggestion,
                createdAt: new Date().toISOString()
            }
        });
    } catch (error) {
        res.status(400).json({
            status: 'fail',
            message: 'Terjadi kesalahan dalam melakukan prediksi'
        });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});