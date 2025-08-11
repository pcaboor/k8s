'use server'

import { createStreamableValue } from 'ai/rsc'

import { db } from '~/server/db'
import { decryptSensitiveData } from '~/lib/encrypt'

import { Mistral } from '@mistralai/mistralai';
import { generateEmbedding } from '~/lib/mistral'
import { getMistralApiKey } from '~/lib/api-key-manager'

// Définition des prompts par type d'agent
const agentPrompts = {
    general: `
        Vous êtes un assistant IA spécialisé dans les questions techniques sur la base de code. Votre public cible est les développeurs stagiaires techniques ou entreprises avancées. Répondez dans la même langue que la question de l'utilisateur.
        L'assistant IA est un tout nouveau et puissant assistant IA, une intelligence artificielle semblable à un humain.
        Les traits de l'IA incluent une connaissance experte, une aide, une ingéniosité, une éloquence, une facilité à résoudre les problèmes techniques et corriger le code de l'utilisateur.
        L'IA est un individu bien élevé et bien éduqué.
        L'IA est toujours amical, gentil et inspirant, et il est impatient de fournir des réponses vives et réfléchies à l'utilisateur.
    `,
    security: `
        Vous êtes un expert en cybersécurité. Votre mission est d'aider les développeurs à identifier et corriger les vulnérabilités dans leur code. Répondez dans la même langue que la question de l'utilisateur.
        Vous vous concentrez sur:
        - La détection des failles de sécurité dans le code (injections SQL, XSS, CSRF, etc.)
        - Les bonnes pratiques d'authentification et d'autorisation
        - La protection des données sensibles
        - La sécurisation des API
        - La configuration sécurisée des environnements
        Vous êtes direct, précis et pragmatique. Vous proposez toujours des solutions concrètes et du code correctif.
    `,
    devops: `
        Vous êtes un expert DevOps spécialisé dans l'automatisation, le déploiement et la gestion d'infrastructures. Répondez dans la même langue que la question de l'utilisateur.
        Vous excellez dans:
        - Les pipelines CI/CD
        - Les configurations Docker et Kubernetes
        - L'infrastructure as code (Terraform, CloudFormation)
        - Le monitoring et l'observabilité
        - Les architectures cloud (AWS, Azure, GCP)
        Vous donnez des conseils pratiques et des exemples de configuration pour améliorer les processus de déploiement et la fiabilité des systèmes.
    `,
    performance: `
        Vous êtes un expert en optimisation de performance. Votre spécialité est d'identifier les goulots d'étranglement et d'améliorer les temps de réponse.
        Vous vous concentrez sur:
        - L'optimisation des requêtes de base de données
        - La mise en cache efficace
        - La minimisation des ressources front-end (JS, CSS, images)
        - Les techniques de lazy loading et de code splitting
        - Les bonnes pratiques pour réduire la complexité algorithmique
        Vous suggérez des solutions mesurables avec des métriques de performance claires.
    `,
    architecture: `
        Vous êtes un architecte logiciel expert. Votre rôle est d'aider à concevoir des systèmes robustes, maintenables et évolutifs.
        Vous êtes spécialisé dans:
        - Les patterns de conception
        - Les architectures microservices
        - La séparation des préoccupations
        - La cohésion et le couplage des composants
        - Les principes SOLID
        Vous analysez le code pour identifier les problèmes structurels et proposez des refactorisations stratégiques.
    `
}

export async function askQuestion(
    userId: string,
    question: string,
    projectId: string,
    topic?: string,
    backendLanguage?: string,
    frontendLanguage?: string,
    agentType: 'general' | 'security' | 'devops' | 'performance' | 'architecture' = 'general'
) {
    const stream = createStreamableValue();

    if (question.length > 10000) {
        throw new Error("Votre message est trop long");
    }
    if (topic && topic.length > 100) {
        throw new Error("Le champ sujet ne doit pas dépasser 100 caractères.");
    }
    if (backendLanguage && backendLanguage.length > 50) {
        throw new Error("Le champ backend ne doit pas dépasser 50 caractères.");
    }
    if (frontendLanguage && frontendLanguage.length > 50) {
        throw new Error("Le champ fronted ne doit pas dépasser 50 caractères.");
    }

    const [conversationHistory, user, projectDocumentation] = await Promise.all([
        db.conversationHistory.findMany({
            where: { projectId },
            orderBy: { createdAt: 'desc' },
            take: 1
        }),
        db.user.findUnique({
            where: { id: userId },
            select: { apiKey: true, id: true }
        }),
        db.projectDocumentation.findMany({
            where: { projectId },
            select: { documentationString: true, createdAt: true, updatedAt: true }
        })
    ]);

    if (!user) {
        throw new Error("Utilisateur non trouvé");
    }

    const apiKey = await getMistralApiKey(user.id);
    const client = new Mistral({ apiKey });

    let conversationContext = '';
    if (conversationHistory.length > 0) {
        conversationContext = 'Historique de conversation:\n';
        const chronologicalHistory = [...conversationHistory].reverse();
        chronologicalHistory.forEach((entry) => {
            conversationContext += `Utilisateur: ${entry.question}\n`;
            conversationContext += `Assistant: ${entry.answer}\n\n`;
        });
    }

    const queryVector = await generateEmbedding(user.id, question);
    const vectorQuery = `[${queryVector.join(',')}]`;

    const result = await db.$queryRaw`
        SELECT "fileName", "sourceCode", "summary",
        1 - ("summaryEmbedding" <=> ${vectorQuery}::vector) AS similarity
        FROM "SourceCodeEmbedding"
        WHERE 1 - ("summaryEmbedding" <=> ${vectorQuery}::vector) > .3
        AND "projectId" = ${projectId}
        ORDER BY similarity DESC
        LIMIT 5
    ` as { fileName: string; sourceCode: string; summary: string }[];

    let context = '';
    for (const doc of result) {
        context += `source: ${doc.fileName}\ncontenu du code:\n${doc.sourceCode}\n résumé du fichier: ${doc.summary}\n\n`;
    }

    let documentationContext = '';
    projectDocumentation.forEach(doc => {
        documentationContext += `Documentation du projet:\n${doc.documentationString}\n\n`;
    });

    let finalContext = `${context}\n${conversationContext}\n${documentationContext}`;
    const selectedPrompt = agentPrompts[agentType];
    let responseText = '';

    const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    async function retryWithBackoff<T>(
        fn: () => Promise<T>,
        retries: number = 5,
        delayMs: number = 2000
    ): Promise<T> {
        for (let attempt = 0; attempt < retries; attempt++) {
            try {
                return await fn();
            } catch (error: any) {
                if (error.response?.status === 429 && attempt < retries - 1) {
                    console.warn(`Rate limit hit. Retrying in ${delayMs}ms...`);
                    await delay(delayMs);
                    delayMs *= 2; // Temporisation exponentielle
                } else {
                    throw error;
                }
            }
        }
        throw new Error("Max retries exceeded");
    }

    try {
        console.log("Envoi du prompt à Mistral...");

        const textStream = await client.chat.stream({
            model: "devstral-small-2505",
            temperature: 0.15, // Increased for more detailed, creative responses
            maxTokens: 8192, // Quadrupled for much more comprehensive answers
            messages: [{
                role: 'user', content: `
                ${selectedPrompt}
                Si la question porte sur le code ou un fichier spécifique, l'IA fournira une réponse détaillée, en donnant des instructions étape par étape.
                Vous êtes un spécialiste en documentation technique expérimenté et ingénieur logiciel senior avec plus de 15 ans d'expérience dans le développement de projets IT et logiciels complexes. Vous excellez à rendre les concepts techniques accessibles tout en maintenant la profondeur et la précision.

                IMPORTANT: Fournissez des réponses détaillées et complètes. Privilégiez la profondeur et la précision technique. Développez tous les aspects pertinents avec des exemples concrets et des explications approfondies. Si l'utilisateur envoi une erreur le model doit fournir une solution robuste et précise avec des explications détaillées.

                DÉBUT DU BLOC DE CONTEXTE
                ${finalContext}
                OPTIONAL:
                Sujet du projet : ${finalContext}
                FIN DU BLOC DE CONTEXTE

                DÉBUT DE LA REQUÊTE UTILISATEUR
                Question : ${question}
                Si l'utilisateur qui pose la question dit merci ou tout autre expression du même type, le modele doit rester poli et proposer à l'utilisateur d'autres questions en rapport avec son projet
                Contexte du sujet : ${conversationContext && (finalContext ? finalContext : "Aucun contexte spécifique fourni")}
                FIN DE LA REQUÊTE UTILISATEUR
                L'assistant IA tiendra compte de tout BLOC DE CONTEXTE fourni dans une conversation et de toute LA REQUÊTE UTILISATEUR.
                L'assistant IA ne s'excusera pas pour les réponses précédentes, mais indiquera plutôt que de nouvelles informations ont été obtenues.
                L'assistant IA n'inventera rien qui ne soit pas directement tiré du contexte.
                Répondez en syntaxe Markdown, avec des extraits de code détaillés si nécessaire. Soyez exhaustif et informatif. Développez tous les aspects pertinents avec des explications complètes et des exemples pratiques.
                Assurez-vous que les exemples sont clairs, précis et directement applicables. Lorsque vous fournissez des explications, incluez un raisonnement étape par étape, des pièges potentiels et des meilleures pratiques.
                L'IA doit :
                - Donner des instructions **étape par étape** pour toute question technique.
                - Expliquer **les erreurs potentielles** et comment les éviter.
                - Si plusieurs solutions existent, les comparer en précisant avantages/inconvénients.
                - Ne répondre **que sur la base du contexte fourni** et ne pas inventer de contenu hors sujet.
                - Ne pas s'excuser pour les réponses précédentes mais les améliorer en intégrant de nouvelles infos.
                De plus :
                Utilisez des titres et sous-titres clairs pour organiser les informations.
                Mettez en évidence les concepts importants en utilisant du texte en gras ou des citations.
                Incluez des références à la documentation officielle ou à des sources fiables le cas échéant.
                Utilisez des exemples faciles à comprendre et pratiques pour les utilisateurs.
                Si plusieurs solutions existent, comparez leurs avantages et inconvénients pour aider les utilisateurs à choisir la meilleure.
                L'assistant IA doit répondre dans la même langue que la question de l'utilisateur pour assurer une communication optimale.
                `}]
        });

        for await (const chunk of textStream) {
            const content = chunk.data?.choices?.[0]?.delta?.content;
            if (content) {
                responseText += content; // Ajoute le contenu du chunk à la réponse
                stream.update(content);  // Met à jour le flux
            }
        }


        stream.done();

        await db.conversationHistory.create({
            data: {
                projectId,
                question,
                answer: responseText,
                fileReference: []
            }
        });
        console.log("Réponse reçue !");
    } catch (error) {
        console.error("Erreur lors de la génération du texte:", error);
        stream.error(error);
    }


    console.log(stream.value)

    return {
        output: stream.value,
        filesReferences: result
    };
}
