// Global teardown for E2E tests
const { chromium } = require('@playwright/test');

async function globalTeardown(config) {
  console.log('🧹 Starting E2E test global teardown...');

  if (process.env.E2E_SETUP_COMPLETE !== 'true') {
    console.log('⚠️  Setup was not completed, skipping teardown');
    return;
  }

  // Create a browser instance for teardown
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    const apiUrl = process.env.API_BASE_URL || 'http://localhost:8000';

    // Clean up test data
    console.log('🗑️  Cleaning up test data...');
    
    try {
      const cleanupResponse = await page.request.post(`${apiUrl}/test/cleanup`, {
        data: {
          action: 'cleanup_test_data',
          cleanupUsers: true,
          cleanupProjects: true,
          cleanupAgents: true,
          cleanupTasks: true,
          cleanupFiles: true
        }
      });
      
      if (cleanupResponse.ok()) {
        console.log('✅ Test data cleanup completed');
      } else {
        console.log('⚠️  Test data cleanup failed, some data may remain');
      }
    } catch (error) {
      console.log('⚠️  Test data cleanup endpoint not available, manual cleanup may be needed');
    }

    // Clean up uploaded test files
    console.log('📁 Cleaning up test files...');
    try {
      const fs = require('fs').promises;
      const path = require('path');
      
      const testUploadsDir = path.join(process.cwd(), 'uploads', 'test');
      try {
        await fs.rmdir(testUploadsDir, { recursive: true });
        console.log('✅ Test files cleaned up');
      } catch (error) {
        if (error.code !== 'ENOENT') {
          console.log('⚠️  Could not clean up test files:', error.message);
        }
      }
    } catch (error) {
      console.log('⚠️  File cleanup error:', error.message);
    }

    // Clean up test artifacts
    console.log('🧹 Cleaning up test artifacts...');
    try {
      const fs = require('fs').promises;
      const path = require('path');
      
      // Clean up temporary screenshots and videos that might be left
      const artifactsDir = path.join(process.cwd(), 'test-results', 'e2e-artifacts');
      try {
        const files = await fs.readdir(artifactsDir);
        const cleanupPromises = files
          .filter(file => file.startsWith('temp-') || file.includes('trace-'))
          .map(file => fs.unlink(path.join(artifactsDir, file)).catch(() => {}));
        
        await Promise.all(cleanupPromises);
        console.log('✅ Test artifacts cleaned up');
      } catch (error) {
        if (error.code !== 'ENOENT') {
          console.log('⚠️  Could not clean up test artifacts:', error.message);
        }
      }
    } catch (error) {
      console.log('⚠️  Artifact cleanup error:', error.message);
    }

    // Reset environment state
    console.log('🔄 Resetting environment state...');
    
    // Clear any cached data
    try {
      const cacheResponse = await page.request.post(`${apiUrl}/test/cache/clear`, {
        data: { action: 'clear_all_caches' }
      });
      
      if (cacheResponse.ok()) {
        console.log('✅ Cache cleared');
      }
    } catch (error) {
      console.log('⚠️  Cache clearing not available');
    }

    // Log test summary if available
    if (process.env.CI) {
      console.log('📊 Test run summary:');
      console.log(`   Environment: ${process.env.CI_ENVIRONMENT_NAME || 'unknown'}`);
      console.log(`   Commit: ${process.env.CI_COMMIT_SHA || 'unknown'}`);
      console.log(`   Branch: ${process.env.CI_COMMIT_REF_NAME || 'unknown'}`);
    }

    console.log('✅ Global teardown completed successfully');

  } catch (error) {
    console.error('❌ Global teardown encountered errors:', error.message);
    // Don't throw here - we want tests to complete even if teardown has issues
  } finally {
    await context.close();
    await browser.close();
    
    // Clear the setup flag
    delete process.env.E2E_SETUP_COMPLETE;
  }
}

module.exports = globalTeardown;