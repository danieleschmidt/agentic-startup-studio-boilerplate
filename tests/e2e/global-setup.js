// Global setup for E2E tests
const { chromium } = require('@playwright/test');

async function globalSetup(config) {
  console.log('🚀 Starting E2E test global setup...');

  // Create a browser instance for setup
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Wait for services to be ready
    console.log('⏳ Waiting for backend API to be ready...');
    const apiUrl = process.env.API_BASE_URL || 'http://localhost:8000';
    
    // Poll the health endpoint until it's ready
    let isApiReady = false;
    let attempts = 0;
    const maxAttempts = 30;
    
    while (!isApiReady && attempts < maxAttempts) {
      try {
        const response = await page.request.get(`${apiUrl}/health`);
        if (response.ok()) {
          console.log('✅ Backend API is ready');
          isApiReady = true;
        } else {
          throw new Error(`API returned status ${response.status()}`);
        }
      } catch (error) {
        attempts++;
        console.log(`⏳ Attempt ${attempts}/${maxAttempts}: API not ready yet, waiting...`);
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
      }
    }

    if (!isApiReady) {
      throw new Error('Backend API failed to become ready within timeout period');
    }

    // Wait for frontend to be ready
    console.log('⏳ Waiting for frontend to be ready...');
    const frontendUrl = process.env.FRONTEND_URL || 'http://localhost:3000';
    
    let isFrontendReady = false;
    attempts = 0;
    
    while (!isFrontendReady && attempts < maxAttempts) {
      try {
        const response = await page.request.get(frontendUrl);
        if (response.ok()) {
          console.log('✅ Frontend is ready');
          isFrontendReady = true;
        } else {
          throw new Error(`Frontend returned status ${response.status()}`);
        }
      } catch (error) {
        attempts++;
        console.log(`⏳ Attempt ${attempts}/${maxAttempts}: Frontend not ready yet, waiting...`);
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
      }
    }

    if (!isFrontendReady) {
      throw new Error('Frontend failed to become ready within timeout period');
    }

    // Setup test data if needed
    console.log('📊 Setting up test data...');
    
    // Create a test user if authentication is enabled
    if (process.env.KEYCLOAK_URL) {
      console.log('👤 Setting up test user in Keycloak...');
      // This would involve creating a test user in Keycloak
      // For now, we'll just log that we would do this
      console.log('⚠️  Keycloak test user setup would be implemented here');
    }

    // Setup test database state
    try {
      const setupResponse = await page.request.post(`${apiUrl}/test/setup`, {
        data: {
          action: 'create_test_data',
          includeUsers: true,
          includeProjects: true,
          includeAgents: true
        }
      });
      
      if (setupResponse.ok()) {
        console.log('✅ Test data setup completed');
      } else {
        console.log('⚠️  Test data setup failed, continuing without pre-populated data');
      }
    } catch (error) {
      console.log('⚠️  Test data setup endpoint not available, continuing without pre-populated data');
    }

    // Store global state for tests
    process.env.E2E_SETUP_COMPLETE = 'true';
    
    console.log('✅ Global setup completed successfully');

  } catch (error) {
    console.error('❌ Global setup failed:', error.message);
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
}

module.exports = globalSetup;