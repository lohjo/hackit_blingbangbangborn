// app/static/js/script.js


document.addEventListener('DOMContentLoaded', function() {
    const topicInput = document.getElementById('topic');
    const ageGroupSelect = document.getElementById('age-group');
    const modelSelect = document.getElementById('model');
    const generateBtn = document.getElementById('generate-btn');
   
    // Model descriptions for better UX
    const modelDescriptions = {
        'flux': {
            text: 'FLUX.1-schnell via HuggingFace - Fast generation with excellent quality. Completely free!',
            badge: 'FREE',
            badgeClass: ''
        },
        'stable-diffusion': {
            text: 'Stable Diffusion XL via HuggingFace - Reliable fallback option. Completely free!',
            badge: 'FREE',
            badgeClass: ''
        },
        'flux-replicate': {
            text: 'FLUX.1-schnell via Replicate - Fastest generation but costs ~$0.003 per image.',
            badge: 'PAID',
            badgeClass: 'paid'
        },
        'openai': {
            text: 'DALL-E 3 - Highest quality but expensive at ~$0.04 per image.',
            badge: 'EXPENSIVE',
            badgeClass: 'expensive'
        }
    };
   
    // Update model description when selection changes
    function updateModelDescription() {
        const selectedModel = modelSelect.value;
        const description = modelDescriptions[selectedModel];
       
        let descriptionDiv = document.querySelector('.model-description');
        if (!descriptionDiv) {
            descriptionDiv = document.createElement('div');
            descriptionDiv.className = 'model-description';
            modelSelect.parentNode.appendChild(descriptionDiv);
        }
       
        descriptionDiv.innerHTML = `
            ${description.text}
            <span class="cost-badge ${description.badgeClass}">${description.badge}</span>
        `;
    }
   
    // Initialize model description
    updateModelDescription();
    modelSelect.addEventListener('change', updateModelDescription);
   
    // Generate button functionality
    generateBtn.addEventListener('click', async function() {
        const topic = topicInput.value.trim();
        const ageGroup = ageGroupSelect.value;
        const model = modelSelect.value;
       
        if (!topic) {
            alert('Please enter a topic.');
            return;
        }


        const originalText = generateBtn.innerHTML;
        const placeholder = document.querySelector('.placeholder-text');
        const image = document.getElementById('generated-image');
       
        // Show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<div class="loading-spinner"></div> Generating...';
       
        if (placeholder) {
            placeholder.style.display = 'flex';
            placeholder.innerHTML = `
                <div class="loading-spinner" style="font-size: 2rem; margin-bottom: 20px;"></div>
                <h3>Generating your educational poster...</h3>
                <p>Using ${modelDescriptions[model].text.split(' - ')[0]}</p>
                <p>This may take a few moments.</p>
            `;
        }
       
        if (image) {
            image.style.display = 'none';
        }


        try {
            // Step 1: Generate poster
            console.log('Starting poster generation...', { topic, ageGroup, model });
           
            const response = await fetch('/generate-poster', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topic: topic,
                    age_group: ageGroup,
                    model: model,
                    complexity: "simple",
                    style: "infographic"
                })
            });


            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
                throw new Error(errorData.detail || `Server error (${response.status})`);
            }


            const data = await response.json();
            console.log('Generation successful:', data);
           
            // Update info panel
            const infoElements = {
                'info-subject': topic,
                'info-date': new Date().toLocaleDateString(),
                'info-resolution': '128x64 (OLED)',
                'info-model': modelDescriptions[model].text.split(' - ')[0],
                'info-cost': data.cost_estimate || modelDescriptions[model].badge,
                'info-time': `${data.generation_time.toFixed(1)}s`
            };
           
            Object.entries(infoElements).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            });


            // Step 2: Display preview image
            const imageUrl = `/poster/${data.image_id}/preview`;
           
            if (image) {
                image.onload = () => {
                    console.log('Image loaded successfully');
                    generateBtn.disabled = false;
                    generateBtn.innerHTML = originalText;
                   
                    if (placeholder) placeholder.style.display = 'none';
                    image.style.display = 'block';
                };
               
                image.onerror = () => {
                    throw new Error('Failed to load generated image');
                };
               
                image.src = imageUrl;
                image.alt = `Educational poster: ${topic} (Age: ${ageGroup})`;
            }
           
            // Add download link for ESP32
            updateDownloadInfo(data.image_id);
           
        } catch (error) {
            console.error('Generation failed:', error);
           
            generateBtn.disabled = false;
            generateBtn.innerHTML = originalText;
           
            if (image) image.style.display = 'none';
           
            if (placeholder) {
                placeholder.style.display = 'flex';
                placeholder.innerHTML = `
                    <i class="fas fa-exclamation-triangle" style="color: #e74c3c;"></i>
                    <h3>Generation Failed</h3>
                    <p>${error.message}</p>
                    <p><small>Please try again or select a different model.</small></p>
                `;
            }
        }
    });
   
    function updateDownloadInfo(imageId) {
        const downloadUrl = `/poster/${imageId}`;
        const previewUrl = `/poster/${imageId}/preview`;
       
        // Add or update download section
        let downloadSection = document.querySelector('.download-section');
        if (!downloadSection) {
            downloadSection = document.createElement('div');
            downloadSection.className = 'download-section';
            downloadSection.style.marginTop = '20px';
            downloadSection.style.padding = '15px';
            downloadSection.style.background = '#e8f5e8';
            downloadSection.style.borderRadius = '8px';
            downloadSection.style.border = '1px solid #4caf50';
           
            const infoPanel = document.querySelector('.image-info');
            if (infoPanel) {
                infoPanel.appendChild(downloadSection);
            }
        }
       
        downloadSection.innerHTML = `
            <h4 style="color: #2e7d32; margin-bottom: 10px;">
                <i class="fas fa-download"></i> ESP32 Ready
            </h4>
            <p style="font-size: 0.9rem; margin-bottom: 10px;">
                Your poster has been converted to 128x64 monochrome bitmap for ESP32 OLED display.
            </p>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <a href="${downloadUrl}" download="${imageId}.xbm"
                style="background: #4caf50; color: white; padding: 8px 12px; border-radius: 5px; text-decoration: none; font-size: 0.85rem;">
                    <i class="fas fa-microchip"></i> Download XBM
                </a>
                <a href="${previewUrl}" target="_blank"
                style="background: #2196f3; color: white; padding: 8px 12px; border-radius: 5px; text-decoration: none; font-size: 0.85rem;">
                    <i class="fas fa-eye"></i> Preview
                </a>
            </div>
        `;
    }
});