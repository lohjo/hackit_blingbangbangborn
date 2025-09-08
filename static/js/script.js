// app/static/js/script.js (or directly in a <script> tag in index.html)

// Add new variables for the select elements
const topicInput = document.getElementById('topic');
const ageGroupSelect = document.getElementById('age-group');
const modelSelect = document.getElementById('model');

// Update generate button functionality
document.getElementById('generate-btn').addEventListener('click', async function() {
    const topic = topicInput.value;
    const ageGroup = ageGroupSelect.value;
    const model = modelSelect.value;
    
    if (!topic.trim()) {
        alert('Please enter a topic.');
        return;
    }

    const generateBtn = this;
    const originalText = generateBtn.innerHTML;
    const placeholder = document.querySelector('.placeholder-text');
    const image = document.getElementById('generated-image');
    
    // Show a loading state
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    placeholder.innerHTML = '<i class="fas fa-spinner fa-spin"></i><h3>Generating your image...</h3><p>This may take a moment.</p>';
    placeholder.style.display = 'flex';
    image.style.display = 'none';

    try {
        // Step 1: Call the /generate-poster API
        const response = await fetch('/generate-poster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic: topic,
                age_group: ageGroup,
                model: model,
                complexity: "simple" // You can add a UI for this later
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate poster.');
        }

        const data = await response.json();
        const imageId = data.image_id;

        // Update info panel
        document.getElementById('info-subject').textContent = topic;
        document.getElementById('info-date').textContent = new Date().toLocaleDateString();
        document.getElementById('info-resolution').textContent = '128x64 (OLED)'; // Correcting for the backend's output

        // Step 2: Call the /poster/{image_id}/preview API to get the image
        const imageUrl = `/poster/${imageId}/preview`;
        
        image.src = imageUrl;
        image.alt = `Generated image for ${topic}`;
        image.style.display = 'block';
        placeholder.style.display = 'none';
        
        // Add a handler for when the image loads
        image.onload = () => {
             // Re-enable button and show success message
            generateBtn.disabled = false;
            generateBtn.innerHTML = originalText;
            console.log('Image successfully generated and loaded.');
        };
        
    } catch (error) {
        console.error('Generation failed:', error);
        placeholder.innerHTML = `<i class="fas fa-exclamation-triangle"></i><h3>Error</h3><p>${error.message}</p>`;
        placeholder.style.display = 'flex';
        image.style.display = 'none';
        generateBtn.disabled = false;
        generateBtn.innerHTML = originalText;
    }
});