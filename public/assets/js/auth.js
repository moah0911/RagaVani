// Local Authentication for RagaVani

// Simple client-side authentication
// This is a simplified version that redirects to the main app
// The actual authentication will be handled by the server-side code

// Tab switching functionality
document.getElementById('login-tab').addEventListener('click', function() {
    document.getElementById('login-tab').classList.add('active');
    document.getElementById('register-tab').classList.remove('active');
    document.getElementById('login-form').classList.add('active');
    document.getElementById('register-form').classList.remove('active');
});

document.getElementById('register-tab').addEventListener('click', function() {
    document.getElementById('register-tab').classList.add('active');
    document.getElementById('login-tab').classList.remove('active');
    document.getElementById('register-form').classList.add('active');
    document.getElementById('login-form').classList.remove('active');
});

// Login functionality
document.getElementById('login-button').addEventListener('click', function() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const errorElement = document.getElementById('login-error');
    const successElement = document.getElementById('login-success');

    errorElement.style.display = 'none';

    if (!email || !password) {
        errorElement.textContent = 'Please enter both email and password.';
        errorElement.style.display = 'block';
        return;
    }

    // Store credentials in localStorage for the main app to use
    localStorage.setItem('ragavani_email', email);
    localStorage.setItem('ragavani_password', password);

    // Show success message
    successElement.style.display = 'block';

    // Redirect to main app
    setTimeout(() => {
        window.location.href = 'https://ragavani.onrender.com';
    }, 2000);
});

// Registration functionality
document.getElementById('register-button').addEventListener('click', function() {
    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;
    const errorElement = document.getElementById('register-error');
    const successElement = document.getElementById('register-success');

    errorElement.style.display = 'none';

    if (!name || !email || !password) {
        errorElement.textContent = 'Please fill in all fields.';
        errorElement.style.display = 'block';
        return;
    }

    if (password !== confirmPassword) {
        errorElement.textContent = 'Passwords do not match.';
        errorElement.style.display = 'block';
        return;
    }

    if (password.length < 6) {
        errorElement.textContent = 'Password must be at least 6 characters long.';
        errorElement.style.display = 'block';
        return;
    }

    // Store registration info in localStorage for the main app to use
    localStorage.setItem('ragavani_register_name', name);
    localStorage.setItem('ragavani_register_email', email);
    localStorage.setItem('ragavani_register_password', password);

    // Show success message
    successElement.style.display = 'block';

    // Redirect to main app
    setTimeout(() => {
        window.location.href = 'https://ragavani.onrender.com';
    }, 2000);
});

// Guest login functionality
document.getElementById('guest-button').addEventListener('click', function() {
    // Set guest flag in localStorage
    localStorage.setItem('ragavani_guest', 'true');

    // Redirect to the app without authentication
    window.location.href = 'https://ragavani.onrender.com';
});

// Check if user has credentials stored
window.addEventListener('load', function() {
    if (localStorage.getItem('ragavani_email') || localStorage.getItem('ragavani_guest')) {
        // User has credentials, redirect to the app
        window.location.href = 'https://ragavani.onrender.com';
    }
});