const form = document.getElementById('process-form');
const controls = document.getElementById('controls');
const status = document.getElementById('status');
const original = document.getElementById('original');
const output = document.getElementById('output');
const download = document.getElementById('download');
let previewUrl;

function clearOutput() {
  output.hidden = true;
  output.removeAttribute('src');
  download.hidden = true;
  download.removeAttribute('href');
  document.getElementById('output-empty').hidden = false;
}

document.getElementById('image').addEventListener('change', (event) => {
  clearOutput();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = event.target.files[0];
  original.hidden = !file;
  document.getElementById('original-empty').hidden = Boolean(file);
  if (file) {
    previewUrl = URL.createObjectURL(file);
    original.src = previewUrl;
  }
  status.dataset.state = 'ready';
  status.textContent = file ? 'Ready to process.' : 'Choose an image to begin.';
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  clearOutput();
  controls.disabled = true;
  form.setAttribute('aria-busy', 'true');
  status.dataset.state = 'working';
  status.textContent = 'Working… The first run loads the model and may fetch VAE/config files. Keep this page open.';
  try {
    const response = await fetch(form.action, { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Processing failed.');
    original.src = result.original;
    original.hidden = false;
    document.getElementById('original-empty').hidden = true;
    output.src = result.output;
    output.hidden = false;
    document.getElementById('output-empty').hidden = true;
    download.href = result.download;
    download.hidden = false;
    status.dataset.state = 'completed';
    status.textContent = `Completed. Seed: ${result.seed}. Your PNG is ready.`;
  } catch (error) {
    status.dataset.state = 'error';
    status.textContent = error.message || 'Could not reach the service. Please try again.';
  } finally {
    controls.disabled = false;
    form.setAttribute('aria-busy', 'false');
  }
});
