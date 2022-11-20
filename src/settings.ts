import joplin from 'api';
import { SettingItemType } from 'api/types';


export interface JarvisSettings {
    openai_api_key: string;
    model: string;
    temperature: number;
    max_tokens: number;
    top_p: number;
    frequency_penalty: number;
    presence_penalty: number;
}

export async function get_settings(): Promise<JarvisSettings> {
    return {
        openai_api_key: await joplin.settings.value('openai_api_key'),
        model: await joplin.settings.value('model'),
        temperature: (await joplin.settings.value('temperature')) / 10,
        max_tokens: await joplin.settings.value('max_tokens'),
        top_p: (await joplin.settings.value('top_p')) / 100,
        frequency_penalty: (await joplin.settings.value('frequency_penalty')) / 10,
        presence_penalty: (await joplin.settings.value('presence_penalty')) / 10,
    }
}

export async function register_settings() {
    await joplin.settings.registerSection('jarvis', {
        label: 'Jarvis',
        iconName: 'fas fa-robot',
    });

    await joplin.settings.registerSettings({
        'openai_api_key': {
            value: '',
            type: SettingItemType.String,
            secure: true,
            section: 'jarvis',
            public: true,
            label: 'OpenAI API Key',
            description: 'Your OpenAI API Key',
        },
        'model': {
            value: 'text-davinci-002',
            type: SettingItemType.String,
            isEnum: true,
            section: 'jarvis',
            public: true,
            label: 'Model',
            description: 'The model to use for asking Jarvis',
            options: {
                'text-davinci-002': 'text-davinci-002',
                'text-curie-001': 'text-curie-001',
                'text-babbage-001': 'text-babbage-001',
                'text-ada-001': 'text-ada-001',
            }
        },
        'temperature': {
            value: 0.7,
            type: SettingItemType.Int,
            minimum: 0,
            maximum: 10,
            step: 1,
            section: 'jarvis',
            public: true,
            label: 'Temperature',
            description: 'The temperature of the model. 0 is the least creative. 10 is the most creative. Higher values produce more creative results, but can also result in more nonsensical text.',
        },
        'max_tokens': {
            value: 256,
            type: SettingItemType.Int,
            minimum: 16,
            maximum: 4096,
            step: 16,
            section: 'jarvis',
            public: true,
            label: 'Max Tokens',
            description: 'The maximum number of tokens to generate. Higher values will result in more text, but can also result in more nonsensical text.',
        },
        'top_p': {
            value: 100,
            type: SettingItemType.Int,
            minimum: 0,
            maximum: 100,
            step: 1,
            section: 'jarvis',
            public: true,
            label: 'Top P',
            description: 'An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p (between 0 and 100) probability mass. So 10 means only the tokens comprising the top 10% probability mass are considered.',
        },
        'frequency_penalty': {
            value: 0,
            type: SettingItemType.Int,
            minimum: -20,
            maximum: 20,
            step: 1,
            section: 'jarvis',
            public: true,
            label: 'Frequency Penalty',
            description: 'A value between -20 and 20 that penalizes new tokens based on whether they appear in the text so far. Can add diversity by preventing the model from repeating the same line verbatim.',
        },
        'presence_penalty': {
            value: 0,
            type: SettingItemType.Int,
            minimum: -20,
            maximum: 20,
            step: 1,
            section: 'jarvis',
            public: true,
            label: 'Presence Penalty',
            description: 'A value between -20 and 20 that penalizes new tokens based on whether they appear in the text so far. Can add diversity by preventing the model from repeating the same line verbatim.',
        },
    });
}
