const { IgApiClient } = require('instagram-private-api');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

class InstagramClothingBot {
    constructor() {
        this.ig = new IgApiClient();
        this.processedMessages = new Set();
        this.aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:5000';
        this.frontendUrl = process.env.FRONTEND_URL || 'http://localhost:8080';
        
        // Clear old processed messages every hour to prevent memory buildup
        setInterval(() => {
            this.processedMessages.clear();
            console.log('🧹 Cleared processed messages cache');
        }, 3600000); // 1 hour
    }

    async login() {
        try {
            // Generate device info with better simulation
            this.ig.state.generateDevice(process.env.INSTAGRAM_USERNAME);
            
            // Add some delay to seem more human
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Try to load existing session first
            try {
                await this.ig.state.deserialize(process.env.INSTAGRAM_USERNAME + '_session.json');
                console.log('✅ Loaded existing session');
                return true;
            } catch (e) {
                console.log('📝 No existing session found, logging in...');
            }
            
            // Login with retry mechanism
            let attempts = 0;
            const maxAttempts = 3;
            
            while (attempts < maxAttempts) {
                try {
                    console.log(`🔄 Login attempt ${attempts + 1}/${maxAttempts}...`);
                    await this.ig.account.login(process.env.INSTAGRAM_USERNAME, process.env.INSTAGRAM_PASSWORD);
                    
                    // Save session for future use
                    await this.ig.state.serialize(process.env.INSTAGRAM_USERNAME + '_session.json');
                    
                    console.log('✅ Successfully logged into Instagram!');
                    return true;
                } catch (error) {
                    attempts++;
                    if (error.message.includes('challenge_required')) {
                        console.log('⚠️  Instagram requires verification. Please:');
                        console.log('   1. Login to Instagram on your phone/computer');
                        console.log('   2. Complete any verification steps');
                        console.log('   3. Wait 10-15 minutes');
                        console.log('   4. Try again');
                        return false;
                    }
                    if (attempts < maxAttempts) {
                        console.log(`⏳ Waiting 30 seconds before retry...`);
                        await new Promise(resolve => setTimeout(resolve, 30000));
                    }
                }
            }
            
            throw new Error('Max login attempts reached');
            
        } catch (error) {
            console.error('❌ Login failed:', error.message);
            console.log('💡 Solutions:');
            console.log('   1. Login to Instagram manually first');
            console.log('   2. Use a fresh test account');
            console.log('   3. Wait 15+ minutes and try again');
            return false;
        }
    }

    async downloadMedia(mediaUrl, filename) {
        try {
            const response = await axios.get(mediaUrl, { responseType: 'stream' });
            const filepath = path.join(__dirname, 'downloads', filename);
            
            // Ensure downloads directory exists
            if (!fs.existsSync(path.join(__dirname, 'downloads'))) {
                fs.mkdirSync(path.join(__dirname, 'downloads'));
            }

            const writer = fs.createWriteStream(filepath);
            response.data.pipe(writer);

            return new Promise((resolve, reject) => {
                writer.on('finish', () => resolve(filepath));
                writer.on('error', reject);
            });
        } catch (error) {
            console.error('Error downloading media:', error);
            return null;
        }
    }

    async processImage(imagePath) {
        try {
            // Use axios with form-data for file upload
            const FormData = require('form-data');
            const formData = new FormData();
            
            formData.append('image', fs.createReadStream(imagePath), {
                filename: path.basename(imagePath),
                contentType: 'image/jpeg'
            });

            const response = await axios.post(`${this.aiServiceUrl}/search`, formData, {
                headers: {
                    ...formData.getHeaders(),
                },
            });

            return response.data;
        } catch (error) {
            console.error('Error processing image with AI service:', error);
            return null;
        }
    }


    async sendResponse(threadId, result) {
        try {
            if (result.exactMatch) {
                const message = `🎉 Perfect! I found the exact item you're looking for!\n\n✨ ${result.product.name}\n💰 Price: $${result.product.price}\n📦 In stock and ready to ship!\n\n🚚 We can deliver this to you in 30 minutes!\n🛒 Order now: ${this.frontendUrl}/product/${result.product.id}\n\nNeed any help with sizing or have questions? Just ask! 😊`;
                await this.ig.entity.directThread(threadId).broadcastText(message);
            } else if (result.similarItems && result.similarItems.length > 0) {
                let message = `🔍 I couldn't find the exact item, but I found some amazing alternatives that are very similar:\n\n`;
                result.similarItems.forEach((item, index) => {
                    const similarity = Math.round(item.similarity * 100);
                    message += `${index + 1}. ✨ ${item.name}\n   💰 $${item.price} (${similarity}% match)\n   🛒 Order: ${this.frontendUrl}/product/${item.id}\n\n`;
                });
                message += `🚚 All items are in stock and can be delivered in 30 minutes!\n\nWhich one catches your eye? Or would you like me to search for something more specific? 😊`;
                await this.ig.entity.directThread(threadId).broadcastText(message);
            } else {
                const message = `🤔 I couldn't find anything similar in our current inventory, but don't worry!\n\n✨ We can custom-order this exact item for you!\n📅 Delivery: 2-3 days\n💰 Competitive pricing guaranteed\n\n🛒 Browse our full catalog: ${this.frontendUrl}\n📝 Or reply "YES" and I'll set up your custom order!\n\nOr send me another photo - maybe I can find something even better! 😊`;
                await this.ig.entity.directThread(threadId).broadcastText(message);
            }
        } catch (error) {
            console.error('Error sending response:', error);
        }
    }

    async handleMessage(thread, message) {
        const messageId = message.id || message.pk || `${thread.thread_id}_${message.timestamp}`;
        
        // Double-check we haven't processed this message
        if (this.processedMessages.has(messageId)) {
            console.log(`⏭️ Already processed message: ${messageId}`);
            return;
        }
        
        // Mark as processed immediately to prevent duplicate processing
        this.processedMessages.add(messageId);

        const sender = thread.users[0];
        
        // Additional check: don't process messages older than 30 seconds
        const messageTime = new Date(parseInt(message.timestamp) * 1000);
        const now = new Date();
        const secondsDiff = (now - messageTime) / 1000;
        
        if (secondsDiff > 30) {
            return;
        }
        
        console.log(`📨 Processing message from @${sender.username}: ${message.text || '[Media]'}`);

        try {
            // Handle text messages
            if (message.text) {
                const text = message.text.toLowerCase();
                console.log(`🔍 Processing text: "${text}"`);
                
                if (text.includes('hi') || text.includes('hello') || text.includes('start') || text.includes('hey')) {
                    console.log(`✅ Sending welcome message...`);
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '👋 Hey there! I\'m your AI fashion assistant! 😊\n\n📸 Just send me a photo of any clothing item you want to find, and I\'ll:\n✨ Find exact matches in our inventory\n🔄 Show you similar alternatives\n🚚 Get it delivered to you in 30 minutes!\n\nReady to find your perfect outfit? Send me a pic! 📱'
                    );
                    console.log(`✅ Welcome message sent!`);
                } else if (text === 'yes' || text.includes('yes') || text.includes('proceed')) {
                    console.log(`✅ Sending custom order message...`);
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        `🎉 Awesome! I'm so excited to help you get that perfect item! 💕\n\n📝 I'll set up your custom order right now.\n📅 Expected delivery: 2-3 days\n💰 You'll get the best price guaranteed!\n\n🛒 Complete your order here: ${this.frontendUrl}/order/custom\n\nNeed any help with sizing or have questions? I'm here for you! 😊`
                    );
                    console.log(`✅ Custom order message sent!`);
                } else if (text.includes('thank') || text.includes('thanks')) {
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '😊 You\'re so welcome! I love helping you find amazing fashion! 💕\n\nGot another item you\'re looking for? Just send me a photo! 📸'
                    );
                } else if (text.includes('help') || text.includes('how')) {
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '🤗 I\'m here to help you find any clothing item you want!\n\n📸 How it works:\n1. Send me a photo of clothing you like\n2. I\'ll search our inventory for matches\n3. Show you exact or similar items\n4. Get it delivered in 30 minutes!\n\n✨ I can find dresses, shirts, pants, shoes, accessories - anything!\n\nReady to start shopping? Send me a pic! 😊'
                    );
                } else {
                    console.log(`❓ Unknown text message: "${text}"`);
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '🤔 I didn\'t quite understand that, but no worries!\n\n📸 Just send me a photo of any clothing item you want to find, and I\'ll help you locate it in our inventory!\n\nOr type "help" if you need guidance! 😊'
                    );
                }
            }

            // Handle media messages
            if (message.media_share || message.media) {
                let mediaUrl = null;
                let filename = null;

                if (message.media_share) {
                    // Handle shared posts/reels
                    const media = message.media_share;
                    if (media.video_versions && media.video_versions.length > 0) {
                        mediaUrl = media.video_versions[0].url;
                        filename = `reel_${Date.now()}.mp4`;
                    } else if (media.image_versions2 && media.image_versions2.candidates.length > 0) {
                        mediaUrl = media.image_versions2.candidates[0].url;
                        filename = `post_${Date.now()}.jpg`;
                    }
                } else if (message.media) {
                    // Handle direct media uploads
                    const media = message.media;
                    
                    if (media.video && media.video.video_url) {
                        mediaUrl = media.video.video_url;
                        filename = `video_${Date.now()}.mp4`;
                    } else if (media.image && media.image.image_url) {
                        mediaUrl = media.image.image_url;
                        filename = `image_${Date.now()}.jpg`;
                    } else if (media.image_versions2 && media.image_versions2.candidates && media.image_versions2.candidates.length > 0) {
                        mediaUrl = media.image_versions2.candidates[0].url;
                        filename = `image_${Date.now()}.jpg`;
                    } else if (media.video_versions && media.video_versions.length > 0) {
                        mediaUrl = media.video_versions[0].url;
                        filename = `video_${Date.now()}.mp4`;
                    }
                }

                if (mediaUrl) {
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '🤖 Ooh, I love this! Let me analyze this image for you... ✨\n\n🔍 Scanning for clothing items...'
                    );

                    const downloadedPath = await this.downloadMedia(mediaUrl, filename);
                    if (downloadedPath) {
                        await this.ig.entity.directThread(thread.thread_id).broadcastText(
                            `👁️ I can see some great clothing items in this image! Let me search our inventory for matches... 🔍`
                        );
                        
                        const result = await this.processImage(downloadedPath);
                        if (result) {
                            await this.sendResponse(thread.thread_id, result);
                        } else {
                            await this.ig.entity.directThread(thread.thread_id).broadcastText(
                                '😔 Oops! I had trouble processing that image. Could you try sending a clearer photo? I want to help you find the perfect item! 📸'
                            );
                        }
                    } else {
                        await this.ig.entity.directThread(thread.thread_id).broadcastText(
                            '😔 I couldn\'t download that image. Could you try uploading it again? I\'m excited to help you find something amazing! 📸'
                        );
                    }
                } else {
                    await this.ig.entity.directThread(thread.thread_id).broadcastText(
                        '😔 I couldn\'t access that media. Could you try uploading the image directly to our chat? I\'m ready to help you find something great! 📸'
                    );
                }
            }
        } catch (error) {
            console.error('Error handling message:', error);
            await this.ig.entity.directThread(thread.thread_id).broadcastText(
                '😔 Oops! Something went wrong on my end. But don\'t worry - I\'m still here to help! 💕\n\nCould you try sending that image again? I\'m excited to help you find something amazing! 📸'
            );
        }
    }

    async checkMessages() {
        try {
            const threads = await this.ig.feed.directInbox().items();
            
            for (const thread of threads) {
                const messages = thread.items;
                if (messages.length > 0) {
                    const latestMessage = messages[0];
                    
                    // Skip if already processed
                    const messageId = latestMessage.id || latestMessage.pk || `${thread.thread_id}_${latestMessage.timestamp}`;
                    if (this.processedMessages.has(messageId)) {
                        continue;
                    }
                    
                    // Skip old messages (older than 1 minute)
                    const messageTime = new Date(parseInt(latestMessage.timestamp) * 1000);
                    const now = new Date();
                    const minutesDiff = (now - messageTime) / (1000 * 60);
                    
                    if (minutesDiff > 1) {
                        continue;
                    }
                    
                    // Skip messages from bot itself
                    const sender = thread.users[0];
                    if (sender.username === process.env.INSTAGRAM_USERNAME) {
                        continue;
                    }
                    
                    // Skip bot status messages
                    if (latestMessage.text && (
                        latestMessage.text.includes('Processing your image') ||
                        latestMessage.text.includes('Please wait') ||
                        latestMessage.text.includes('Found exact match') ||
                        latestMessage.text.includes('We don\'t have the exact') ||
                        latestMessage.text.includes('We couldn\'t find') ||
                        latestMessage.text.includes('Ooh, I love this') ||
                        latestMessage.text.includes('Scanning for clothing') ||
                        latestMessage.text.includes('I can see some great') ||
                        latestMessage.text.includes('Perfect! I found') ||
                        latestMessage.text.includes('I couldn\'t find the exact') ||
                        latestMessage.text.includes('I couldn\'t find anything similar')
                    )) {
                        continue;
                    }
                    
                    await this.handleMessage(thread, latestMessage);
                }
            }
        } catch (error) {
            console.error('Error checking messages:', error);
        }
    }

    async start() {
        console.log('🚀 Starting Instagram Clothing Bot...');
        
        const loginSuccess = await this.login();
        if (!loginSuccess) {
            console.log('❌ Failed to login. Please check your credentials in .env file');
            return;
        }

        console.log('✅ Bot is running! Checking for messages every 5 seconds...');
        console.log('📱 Send a DM to your bot account to test it!');
        
        // Check messages every 5 seconds for better responsiveness
        setInterval(() => {
            this.checkMessages();
        }, 5000);
    }
}

// Start the bot
const bot = new InstagramClothingBot();
bot.start().catch(console.error);