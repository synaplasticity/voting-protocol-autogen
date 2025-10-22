"""
MemGPT Multi-Agent Auction System with A2A Communication
========================================================

This example demonstrates a multi-agent auction system where:
1. Bidding agents propose prices for services
2. An independent auction agent selects the best bid
3. Uses A2A (Agent-to-Agent) protocol for communication
4. Extensible selection mechanism
5. Configurable LLM models (default: Gemini)
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

# MemGPT/Letta imports (simulated structure)
from letta import create_agent, AgentState
from letta.schemas import LLMConfig
from letta.client import create_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Data Models and Enums
# ==============================================================================

class ServiceType(Enum):
    WEB_DEVELOPMENT = "web_development"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_WRITING = "content_writing"
    GRAPHIC_DESIGN = "graphic_design"
    CONSULTING = "consulting"

class MessageType(Enum):
    BID_REQUEST = "bid_request"
    BID_RESPONSE = "bid_response"
    AUCTION_RESULT = "auction_result"
    AGENT_REGISTRATION = "agent_registration"

@dataclass
class Service:
    """Represents a service being auctioned"""
    service_id: str
    service_type: ServiceType
    description: str
    requirements: Dict[str, Any]
    deadline: str
    budget_range: tuple  # (min, max)

@dataclass
class Bid:
    """Represents a bid from an agent"""
    bid_id: str
    agent_id: str
    service_id: str
    price: float
    delivery_time_days: int
    quality_score: float  # 0-10 scale
    additional_services: List[str]
    timestamp: str
    confidence: float  # 0-1 scale

@dataclass
class A2AMessage:
    """A2A Protocol Message Structure"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: str
    reply_to: Optional[str] = None

# ==============================================================================
# Selection Strategies (Extensible)
# ==============================================================================

class SelectionStrategy(ABC):
    """Abstract base class for bid selection strategies"""
    
    @abstractmethod
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        """Select the winning bid based on the strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy"""
        pass

class LowestPriceStrategy(SelectionStrategy):
    """Selects the bid with the lowest price"""
    
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        if not bids:
            return None
        return min(bids, key=lambda bid: bid.price)
    
    def get_strategy_name(self) -> str:
        return "Lowest Price"

class ValueBasedStrategy(SelectionStrategy):
    """Selects based on value = quality_score / (price * delivery_time)"""
    
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        if not bids:
            return None
        
        def calculate_value(bid: Bid) -> float:
            # Higher quality_score is better, lower price and delivery_time is better
            # Add small epsilon to avoid division by zero
            return bid.quality_score / ((bid.price + 1) * (bid.delivery_time_days + 1))
        
        return max(bids, key=calculate_value)
    
    def get_strategy_name(self) -> str:
        return "Value-Based"

class WeightedScoreStrategy(SelectionStrategy):
    """Selects based on weighted score of multiple factors"""
    
    def __init__(self, price_weight=0.4, quality_weight=0.3, time_weight=0.2, confidence_weight=0.1):
        self.price_weight = price_weight
        self.quality_weight = quality_weight
        self.time_weight = time_weight
        self.confidence_weight = confidence_weight
    
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        if not bids:
            return None
        
        # Normalize factors
        max_price = max(bid.price for bid in bids)
        min_price = min(bid.price for bid in bids)
        max_time = max(bid.delivery_time_days for bid in bids)
        min_time = min(bid.delivery_time_days for bid in bids)
        
        def calculate_weighted_score(bid: Bid) -> float:
            # Normalize price (lower is better) - invert scale
            price_score = 1 - (bid.price - min_price) / (max_price - min_price + 1)
            
            # Quality score (higher is better) - already 0-10, normalize to 0-1
            quality_score = bid.quality_score / 10.0
            
            # Time score (lower is better) - invert scale
            time_score = 1 - (bid.delivery_time_days - min_time) / (max_time - min_time + 1)
            
            # Confidence score (already 0-1)
            confidence_score = bid.confidence
            
            return (price_score * self.price_weight + 
                   quality_score * self.quality_weight + 
                   time_score * self.time_weight + 
                   confidence_score * self.confidence_weight)
        
        return max(bids, key=calculate_weighted_score)
    
    def get_strategy_name(self) -> str:
        return f"Weighted Score (P:{self.price_weight}, Q:{self.quality_weight}, T:{self.time_weight}, C:{self.confidence_weight})"

# ==============================================================================
# A2A Communication Protocol
# ==============================================================================

class A2ACommunicationProtocol:
    """Handles Agent-to-Agent communication"""
    
    def __init__(self):
        self.message_queue: Dict[str, List[A2AMessage]] = {}
        self.agents: Dict[str, 'BaseAgent'] = {}
    
    def register_agent(self, agent: 'BaseAgent'):
        """Register an agent for communication"""
        self.agents[agent.agent_id] = agent
        self.message_queue[agent.agent_id] = []
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def send_message(self, message: A2AMessage):
        """Send a message to the target agent"""
        if message.receiver_id not in self.message_queue:
            logger.error(f"Agent {message.receiver_id} not found")
            return False
        
        self.message_queue[message.receiver_id].append(message)
        logger.info(f"Message sent from {message.sender_id} to {message.receiver_id}: {message.message_type}")
        
        # Notify the receiver agent
        if message.receiver_id in self.agents:
            await self.agents[message.receiver_id].receive_message(message)
        
        return True
    
    def get_messages(self, agent_id: str) -> List[A2AMessage]:
        """Get messages for an agent"""
        return self.message_queue.get(agent_id, [])
    
    def clear_messages(self, agent_id: str):
        """Clear messages for an agent"""
        if agent_id in self.message_queue:
            self.message_queue[agent_id] = []

# ==============================================================================
# Base Agent Class
# ==============================================================================

class BaseAgent:
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, llm_config: LLMConfig, communication_protocol: A2ACommunicationProtocol):
        self.agent_id = agent_id
        self.llm_config = llm_config
        self.communication_protocol = communication_protocol
        self.letta_client = create_client()
        self.agent_state = None
        communication_protocol.register_agent(self)
    
    async def initialize(self):
        """Initialize the MemGPT agent"""
        self.agent_state = create_agent(
            name=self.agent_id,
            llm_config=self.llm_config
        )
        logger.info(f"Initialized agent: {self.agent_id}")
    
    async def receive_message(self, message: A2AMessage):
        """Handle incoming A2A messages"""
        logger.info(f"Agent {self.agent_id} received message: {message.message_type}")
        await self.process_message(message)
    
    async def send_message(self, receiver_id: str, message_type: MessageType, payload: Dict[str, Any], reply_to: str = None):
        """Send a message to another agent"""
        message = A2AMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now().isoformat(),
            reply_to=reply_to
        )
        await self.communication_protocol.send_message(message)
    
    @abstractmethod
    async def process_message(self, message: A2AMessage):
        """Process incoming messages - to be implemented by subclasses"""
        pass

# ==============================================================================
# Bidding Agent
# ==============================================================================

class BiddingAgent(BaseAgent):
    """Agent that submits bids for services"""
    
    def __init__(self, agent_id: str, llm_config: LLMConfig, communication_protocol: A2ACommunicationProtocol, 
                 specialties: List[ServiceType], base_rate: float = 50.0):
        super().__init__(agent_id, llm_config, communication_protocol)
        self.specialties = specialties
        self.base_rate = base_rate
        self.active_bids: Dict[str, Bid] = {}
    
    async def process_message(self, message: A2AMessage):
        """Process incoming messages"""
        if message.message_type == MessageType.BID_REQUEST:
            await self.handle_bid_request(message)
        elif message.message_type == MessageType.AUCTION_RESULT:
            await self.handle_auction_result(message)
    
    async def handle_bid_request(self, message: A2AMessage):
        """Handle bid request from auction agent"""
        service_data = message.payload.get('service')
        if not service_data:
            return
        
        service = Service(**service_data)
        
        # Check if we can handle this service type
        if service.service_type not in self.specialties:
            logger.info(f"Agent {self.agent_id} cannot handle {service.service_type}")
            return
        
        # Generate bid using LLM
        bid = await self.generate_bid(service)
        
        if bid:
            self.active_bids[service.service_id] = bid
            
            # Send bid response
            await self.send_message(
                receiver_id=message.sender_id,
                message_type=MessageType.BID_RESPONSE,
                payload={'bid': asdict(bid)},
                reply_to=message.message_id
            )
    
    async def generate_bid(self, service: Service) -> Optional[Bid]:
        """Generate a bid using the LLM"""
        try:
            # Prepare context for LLM
            context = f"""
            You are a service provider bidding on a project. Here are the details:
            
            Service: {service.service_type.value}
            Description: {service.description}
            Requirements: {json.dumps(service.requirements, indent=2)}
            Deadline: {service.deadline}
            Budget Range: ${service.budget_range[0]} - ${service.budget_range[1]}
            
            Your base rate is ${self.base_rate}/hour.
            Your specialties: {[s.value for s in self.specialties]}
            
            Generate a competitive bid considering:
            1. Price (be competitive but profitable)
            2. Delivery time in days
            3. Your quality score (0-10, be realistic about your capabilities)
            4. Any additional services you can provide
            5. Your confidence level (0-1) in completing this project
            
            Respond with a JSON object containing:
            - price: float
            - delivery_time_days: int
            - quality_score: float (0-10)
            - additional_services: list of strings
            - confidence: float (0-1)
            """
            
            # Use MemGPT to generate response
            response = await self.letta_client.send_message(
                agent_id=self.agent_state.id,
                message=context,
                role="user"
            )
            
            # Parse LLM response (simplified - in practice you'd need better parsing)
            # For this example, we'll generate a sample bid
            import random
            
            # Calculate price based on service complexity and budget range
            complexity_factor = len(service.requirements) * 0.1 + 1.0
            price = min(
                service.budget_range[1],
                max(service.budget_range[0], self.base_rate * complexity_factor * random.uniform(0.8, 1.2))
            )
            
            bid = Bid(
                bid_id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                service_id=service.service_id,
                price=round(price, 2),
                delivery_time_days=random.randint(3, 14),
                quality_score=random.uniform(7.0, 9.5),
                additional_services=[
                    "24/7 Support", "Free Revisions", "Documentation"
                ][:random.randint(1, 3)],
                timestamp=datetime.now().isoformat(),
                confidence=random.uniform(0.7, 0.95)
            )
            
            logger.info(f"Agent {self.agent_id} generated bid: ${bid.price} for service {service.service_id}")
            return bid
            
        except Exception as e:
            logger.error(f"Error generating bid for agent {self.agent_id}: {e}")
            return None
    
    async def handle_auction_result(self, message: A2AMessage):
        """Handle auction result notification"""
        result = message.payload
        service_id = result.get('service_id')
        winner_id = result.get('winner_id')
        
        if service_id in self.active_bids:
            if winner_id == self.agent_id:
                logger.info(f"🎉 Agent {self.agent_id} WON auction for service {service_id}!")
            else:
                logger.info(f"Agent {self.agent_id} did not win auction for service {service_id}")
            
            # Clean up
            del self.active_bids[service_id]

# ==============================================================================
# Auction Agent
# ==============================================================================

class AuctionAgent(BaseAgent):
    """Independent agent that manages auctions and selects winners"""
    
    def __init__(self, agent_id: str, llm_config: LLMConfig, communication_protocol: A2ACommunicationProtocol,
                 selection_strategy: SelectionStrategy = None):
        super().__init__(agent_id, llm_config, communication_protocol)
        self.selection_strategy = selection_strategy or LowestPriceStrategy()
        self.active_auctions: Dict[str, Dict] = {}
        self.bidding_agents: List[str] = []
    
    def set_selection_strategy(self, strategy: SelectionStrategy):
        """Set or change the selection strategy"""
        self.selection_strategy = strategy
        logger.info(f"Auction agent strategy changed to: {strategy.get_strategy_name()}")
    
    def register_bidding_agent(self, agent_id: str):
        """Register a bidding agent"""
        if agent_id not in self.bidding_agents:
            self.bidding_agents.append(agent_id)
            logger.info(f"Registered bidding agent: {agent_id}")
    
    async def start_auction(self, service: Service, auction_duration_seconds: int = 30):
        """Start an auction for a service"""
        auction_id = str(uuid.uuid4())
        
        self.active_auctions[service.service_id] = {
            'auction_id': auction_id,
            'service': service,
            'bids': [],
            'start_time': datetime.now(),
            'duration': auction_duration_seconds
        }
        
        logger.info(f"Starting auction for service: {service.service_id}")
        
        # Send bid requests to all registered bidding agents
        for agent_id in self.bidding_agents:
            await self.send_message(
                receiver_id=agent_id,
                message_type=MessageType.BID_REQUEST,
                payload={'service': asdict(service), 'auction_id': auction_id}
            )
        
        # Wait for bids
        await asyncio.sleep(auction_duration_seconds)
        
        # Select winner
        await self.finalize_auction(service.service_id)
    
    async def process_message(self, message: A2AMessage):
        """Process incoming messages"""
        if message.message_type == MessageType.BID_RESPONSE:
            await self.handle_bid_response(message)
    
    async def handle_bid_response(self, message: A2AMessage):
        """Handle bid responses from bidding agents"""
        bid_data = message.payload.get('bid')
        if not bid_data:
            return
        
        bid = Bid(**bid_data)
        service_id = bid.service_id
        
        if service_id in self.active_auctions:
            self.active_auctions[service_id]['bids'].append(bid)
            logger.info(f"Received bid from {bid.agent_id}: ${bid.price} for service {service_id}")
    
    async def finalize_auction(self, service_id: str):
        """Finalize auction and select winner"""
        if service_id not in self.active_auctions:
            return
        
        auction_data = self.active_auctions[service_id]
        bids = auction_data['bids']
        service = auction_data['service']
        
        logger.info(f"Finalizing auction for service {service_id} with {len(bids)} bids")
        
        # Select winner using the configured strategy
        winning_bid = self.selection_strategy.select_winner(bids, service)
        
        result = {
            'service_id': service_id,
            'total_bids': len(bids),
            'selection_strategy': self.selection_strategy.get_strategy_name(),
            'winner_id': winning_bid.agent_id if winning_bid else None,
            'winning_price': winning_bid.price if winning_bid else None,
            'all_bids': [asdict(bid) for bid in bids]
        }
        
        # Notify all participants
        for agent_id in self.bidding_agents:
            await self.send_message(
                receiver_id=agent_id,
                message_type=MessageType.AUCTION_RESULT,
                payload=result
            )
        
        # Log results
        if winning_bid:
            logger.info(f"🏆 Auction winner: {winning_bid.agent_id} with bid ${winning_bid.price}")
            logger.info(f"Selection strategy: {self.selection_strategy.get_strategy_name()}")
        else:
            logger.info("No winner selected for auction")
        
        # Clean up
        del self.active_auctions[service_id]
        
        return result

# ==============================================================================
# Configuration and Setup
# ==============================================================================

def create_llm_config(model_name: str = "gemini-1.5-pro") -> LLMConfig:
    """Create LLM configuration"""
    return LLMConfig(
        model=model_name,
        model_endpoint_type="google",
        model_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        context_window=32000,
    )

async def setup_auction_system():
    """Setup the complete auction system"""
    # Initialize communication protocol
    comm_protocol = A2ACommunicationProtocol()
    
    # Create LLM configs for different models
    gemini_config = create_llm_config("gemini-1.5-pro")
    claude_config = create_llm_config("claude-3-sonnet-20240229")  # Alternative model
    
    # Create auction agent with extensible strategy
    auction_agent = AuctionAgent(
        agent_id="auction_coordinator",
        llm_config=gemini_config,
        communication_protocol=comm_protocol,
        selection_strategy=WeightedScoreStrategy()  # Start with weighted strategy
    )
    await auction_agent.initialize()
    
    # Create bidding agents with different specialties and models
    bidding_agents = []
    
    # Web development specialist
    web_agent = BiddingAgent(
        agent_id="web_dev_specialist",
        llm_config=gemini_config,
        communication_protocol=comm_protocol,
        specialties=[ServiceType.WEB_DEVELOPMENT, ServiceType.GRAPHIC_DESIGN],
        base_rate=75.0
    )
    await web_agent.initialize()
    bidding_agents.append(web_agent)
    
    # Data analysis specialist
    data_agent = BiddingAgent(
        agent_id="data_analyst",
        llm_config=claude_config,  # Using different model
        communication_protocol=comm_protocol,
        specialties=[ServiceType.DATA_ANALYSIS, ServiceType.CONSULTING],
        base_rate=85.0
    )
    await data_agent.initialize()
    bidding_agents.append(data_agent)
    
    # Content writing specialist
    content_agent = BiddingAgent(
        agent_id="content_writer",
        llm_config=gemini_config,
        communication_protocol=comm_protocol,
        specialties=[ServiceType.CONTENT_WRITING, ServiceType.CONSULTING],
        base_rate=45.0
    )
    await content_agent.initialize()
    bidding_agents.append(content_agent)
    
    # General purpose agent
    general_agent = BiddingAgent(
        agent_id="generalist",
        llm_config=gemini_config,
        communication_protocol=comm_protocol,
        specialties=list(ServiceType),  # Can handle all service types
        base_rate=60.0
    )
    await general_agent.initialize()
    bidding_agents.append(general_agent)
    
    # Register all bidding agents with the auction agent
    for agent in bidding_agents:
        auction_agent.register_bidding_agent(agent.agent_id)
    
    return auction_agent, bidding_agents, comm_protocol

# ==============================================================================
# Demo and Testing
# ==============================================================================

async def run_auction_demo():
    """Run a demonstration of the auction system"""
    logger.info("🚀 Starting MemGPT Multi-Agent Auction System Demo")
    
    # Setup the system
    auction_agent, bidding_agents, comm_protocol = await setup_auction_system()
    
    # Create sample services to auction
    services = [
        Service(
            service_id="web_001",
            service_type=ServiceType.WEB_DEVELOPMENT,
            description="Build a modern e-commerce website with React and Node.js",
            requirements={
                "framework": "React",
                "backend": "Node.js",
                "database": "MongoDB",
                "payment_integration": True,
                "responsive_design": True,
                "admin_panel": True
            },
            deadline="2024-01-15",
            budget_range=(2000, 5000)
        ),
        Service(
            service_id="data_002",
            service_type=ServiceType.DATA_ANALYSIS,
            description="Analyze customer behavior data and create predictive models",
            requirements={
                "dataset_size": "100k records",
                "analysis_type": "predictive modeling",
                "tools": ["Python", "scikit-learn", "pandas"],
                "visualization": True,
                "report": True
            },
            deadline="2024-01-20",
            budget_range=(1500, 3500)
        ),
        Service(
            service_id="content_003",
            service_type=ServiceType.CONTENT_WRITING,
            description="Create SEO-optimized blog posts and website copy",
            requirements={
                "word_count": "5000 words",
                "seo_optimized": True,
                "target_audience": "B2B",
                "research_required": True,
                "revisions": 2
            },
            deadline="2024-01-10",
            budget_range=(800, 1500)
        )
    ]
    
    # Run auctions with different strategies
    strategies = [
        LowestPriceStrategy(),
        ValueBasedStrategy(),
        WeightedScoreStrategy(price_weight=0.5, quality_weight=0.3, time_weight=0.1, confidence_weight=0.1)
    ]
    
    results = []
    
    for i, service in enumerate(services):
        logger.info(f"\n{'='*60}")
        logger.info(f"AUCTION {i+1}: {service.description}")
        logger.info(f"Service Type: {service.service_type.value}")
        logger.info(f"Budget Range: ${service.budget_range[0]} - ${service.budget_range[1]}")
        
        # Set strategy for this auction
        strategy = strategies[i % len(strategies)]
        auction_agent.set_selection_strategy(strategy)
        
        # Run the auction
        result = await auction_agent.start_auction(service, auction_duration_seconds=10)
        results.append(result)
        
        logger.info(f"Auction completed! Winner: {result['winner_id']}")
        logger.info(f"Strategy used: {result['selection_strategy']}")
        
        # Wait between auctions
        await asyncio.sleep(2)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("AUCTION SUMMARY")
    logger.info(f"{'='*60}")
    
    for i, result in enumerate(results):
        service = services[i]
        logger.info(f"\nService: {service.service_id}")
        logger.info(f"Winner: {result['winner_id']}")
        logger.info(f"Winning Price: ${result['winning_price']}")
        logger.info(f"Total Bids: {result['total_bids']}")
        logger.info(f"Strategy: {result['selection_strategy']}")
    
    logger.info("\n🎉 Auction demo completed!")
    
    return results

# ==============================================================================
# Future Enhancement Examples
# ==============================================================================

class MLBasedSelectionStrategy(SelectionStrategy):
    """Example of an ML-based selection strategy (future enhancement)"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # In practice, you would load a trained ML model here
        self.model = None
    
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        if not bids:
            return None
        
        # Placeholder for ML-based selection
        # In practice, you would:
        # 1. Extract features from bids and service
        # 2. Use trained model to predict success probability
        # 3. Select bid with highest predicted success
        
        # For now, fall back to value-based strategy
        fallback_strategy = ValueBasedStrategy()
        return fallback_strategy.select_winner(bids, service)
    
    def get_strategy_name(self) -> str:
        return "ML-Based Selection"

class MultiCriteriaSelectionStrategy(SelectionStrategy):
    """Example of multi-criteria decision analysis strategy"""
    
    def __init__(self, criteria_weights: Dict[str, float] = None):
        self.criteria_weights = criteria_weights or {
            'price': 0.3,
            'quality': 0.25,
            'delivery_time': 0.2,
            'agent_reputation': 0.15,
            'past_performance': 0.1
        }
    
    def select_winner(self, bids: List[Bid], service: Service) -> Optional[Bid]:
        if not bids:
            return None
        
        # Implement TOPSIS or similar MCDA method
        # This is a simplified version
        scores = {}
        
        for bid in bids:
            score = 0
            # Add scoring logic based on multiple criteria
            # This is where you'd implement sophisticated decision analysis
            score = bid.quality_score * self.criteria_weights['quality']  # Simplified
            scores[bid.bid_id] = score
        
        best_bid_id = max(scores, key=scores.get)
        return next(bid for bid in bids if bid.bid_id == best_bid_id)
    
    def get_strategy_name(self) -> str:
        return "Multi-Criteria Decision Analysis"

# ==============================================================================
# Configuration Management
# ==============================================================================

class AuctionSystemConfig:
    """Configuration management for the auction system"""
    
    def __init__(self):
        self.default_llm_model = "gemini-1.5-pro"
        self.auction_duration = 30  # seconds
        self.max_bids_per_agent = 5
        self.min_bid_confidence = 0.5
        self.selection_strategies = {
            'lowest_price': LowestPriceStrategy(),
            'value_based': ValueBasedStrategy(),
            'weighted_score': WeightedScoreStrategy(),
            'ml_based': MLBasedSelectionStrategy(),
            'multi_criteria': MultiCriteriaSelectionStrategy()
        }
    
    def get_strategy(self, strategy_name: str) -> SelectionStrategy:
        """Get selection strategy by name"""
        return self.selection_strategies.get(strategy_name, self.selection_strategies['lowest_price'])

# ==============================================================================
# Monitoring and Analytics
# ==============================================================================

class AuctionAnalytics:
    """Analytics and monitoring for auction performance"""
    
    def __init__(self):
        self.auction_history: List[Dict] = []
        self.agent_performance: Dict[str, Dict] = {}
    
    def record_auction_result(self, result: Dict):
        """Record auction result for analytics"""
        self.auction_history.append({
            **result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update agent performance metrics
        winner_id = result.get('winner_id')
        if winner_id:
            if winner_id not in self.agent_performance:
                self.agent_performance[winner_id] = {
                    'wins': 0,
                    'total_bids': 0,
                    'total_value': 0,
                    'avg_winning_price': 0
                }
            
            self.agent_performance[winner_id]['wins'] += 1
            self.agent_performance[winner_id]['total_value'] += result.get('winning_price', 0)
            self.agent_performance[winner_id]['avg_winning_price'] = (
                self.agent_performance[winner_id]['total_value'] / 
                self.agent_performance[winner_id]['wins']
            )
        
        # Record participation for all agents
        for bid_data in result.get('all_bids', []):
            agent_id = bid_data.get('agent_id')
            if agent_id:
                if agent_id not in self.agent_performance:
                    self.agent_performance[agent_id] = {
                        'wins': 0,
                        'total_bids': 0,
                        'total_value': 0,
                        'avg_winning_price': 0
                    }
                self.agent_performance[agent_id]['total_bids'] += 1
    
    def get_agent_win_rate(self, agent_id: str) -> float:
        """Calculate agent win rate"""
        if agent_id not in self.agent_performance:
            return 0.0
        
        performance = self.agent_performance[agent_id]
        return performance['wins'] / max(performance['total_bids'], 1)
    
    def get_strategy_performance(self) -> Dict[str, Dict]:
        """Analyze performance by selection strategy"""
        strategy_stats = {}
        
        for auction in self.auction_history:
            strategy = auction.get('selection_strategy', 'unknown')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total_auctions': 0,
                    'avg_winning_price': 0,
                    'avg_bids_per_auction': 0,
                    'total_value': 0
                }
            
            strategy_stats[strategy]['total_auctions'] += 1
            strategy_stats[strategy]['total_value'] += auction.get('winning_price', 0)
            strategy_stats[strategy]['avg_bids_per_auction'] += auction.get('total_bids', 0)
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats['total_auctions'] > 0:
                stats['avg_winning_price'] = stats['total_value'] / stats['total_auctions']
                stats['avg_bids_per_auction'] = stats['avg_bids_per_auction'] / stats['total_auctions']
        
        return strategy_stats

# ==============================================================================
# Advanced Agent Features
# ==============================================================================

class EnhancedBiddingAgent(BiddingAgent):
    """Enhanced bidding agent with learning capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bid_history: List[Dict] = []
        self.win_rate = 0.0
        self.learning_rate = 0.1
        self.risk_tolerance = 0.5  # 0 = risk averse, 1 = risk seeking
    
    async def generate_bid(self, service: Service) -> Optional[Bid]:
        """Enhanced bid generation with learning from history"""
        try:
            # Analyze historical performance
            recent_bids = self.bid_history[-10:]  # Last 10 bids
            if recent_bids:
                avg_win_rate = sum(1 for bid in recent_bids if bid.get('won', False)) / len(recent_bids)
                self.win_rate = avg_win_rate
            
            # Adjust bidding strategy based on performance
            aggressiveness = self._calculate_bidding_aggressiveness()
            
            # Enhanced LLM prompt with learning context
            context = f"""
            You are an experienced service provider with the following profile:
            - Agent ID: {self.agent_id}
            - Specialties: {[s.value for s in self.specialties]}
            - Base rate: ${self.base_rate}/hour
            - Recent win rate: {self.win_rate:.2%}
            - Risk tolerance: {self.risk_tolerance}
            - Bidding aggressiveness: {aggressiveness}
            
            Service Details:
            - Type: {service.service_type.value}
            - Description: {service.description}
            - Requirements: {json.dumps(service.requirements, indent=2)}
            - Deadline: {service.deadline}
            - Budget Range: ${service.budget_range[0]} - ${service.budget_range[1]}
            
            Historical Context:
            {self._get_historical_context()}
            
            Generate a strategic bid considering:
            1. Your historical performance and learning
            2. Market positioning based on win rate
            3. Risk-adjusted pricing
            4. Competitive differentiation
            5. Quality vs. price trade-offs
            
            Adjust your strategy based on your win rate:
            - If win rate is low (<30%), be more aggressive on pricing
            - If win rate is high (>70%), you can afford to bid higher for quality
            - Consider your risk tolerance in pricing decisions
            
            Respond with JSON containing your bid details.
            """
            
            # Generate base bid using parent method logic (simplified for demo)
            base_bid = await super().generate_bid(service)
            
            if base_bid:
                # Apply learning adjustments
                adjusted_price = self._adjust_price_with_learning(base_bid.price, aggressiveness)
                base_bid.price = adjusted_price
                base_bid.confidence *= (0.8 + 0.4 * self.win_rate)  # Adjust confidence based on track record
                
                return base_bid
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced bid generation for {self.agent_id}: {e}")
            return await super().generate_bid(service)
    
    def _calculate_bidding_aggressiveness(self) -> float:
        """Calculate how aggressive to be in bidding based on performance"""
        if self.win_rate < 0.2:
            return 0.8  # Very aggressive (lower prices)
        elif self.win_rate < 0.4:
            return 0.6  # Moderate aggressive
        elif self.win_rate > 0.7:
            return 0.2  # Conservative (higher prices for quality)
        else:
            return 0.4  # Balanced approach
    
    def _adjust_price_with_learning(self, base_price: float, aggressiveness: float) -> float:
        """Adjust price based on learning and aggressiveness"""
        # More aggressive = lower price multiplier
        price_multiplier = 1.0 - (aggressiveness * 0.3)
        return round(base_price * price_multiplier, 2)
    
    def _get_historical_context(self) -> str:
        """Get relevant historical context for bidding"""
        if not self.bid_history:
            return "No historical data available."
        
        recent_wins = [bid for bid in self.bid_history[-5:] if bid.get('won', False)]
        recent_losses = [bid for bid in self.bid_history[-5:] if not bid.get('won', False)]
        
        context = f"Recent performance: {len(recent_wins)} wins, {len(recent_losses)} losses. "
        
        if recent_wins:
            avg_winning_price = sum(bid['price'] for bid in recent_wins) / len(recent_wins)
            context += f"Average winning bid: ${avg_winning_price:.2f}. "
        
        if recent_losses:
            avg_losing_price = sum(bid['price'] for bid in recent_losses) / len(recent_losses)
            context += f"Average losing bid: ${avg_losing_price:.2f}. "
        
        return context
    
    async def handle_auction_result(self, message: A2AMessage):
        """Enhanced result handling with learning"""
        result = message.payload
        service_id = result.get('service_id')
        winner_id = result.get('winner_id')
        
        if service_id in self.active_bids:
            bid = self.active_bids[service_id]
            won = winner_id == self.agent_id
            
            # Record result for learning
            self.bid_history.append({
                'service_id': service_id,
                'price': bid.price,
                'won': won,
                'timestamp': datetime.now().isoformat(),
                'winning_price': result.get('winning_price'),
                'total_bids': result.get('total_bids', 0)
            })
            
            # Log with learning context
            if won:
                logger.info(f"🎉 {self.agent_id} WON with ${bid.price} (win rate: {self.win_rate:.2%})")
            else:
                winning_price = result.get('winning_price', 'unknown')
                logger.info(f"{self.agent_id} lost with ${bid.price} vs winning ${winning_price}")
            
            # Clean up
            del self.active_bids[service_id]

# ==============================================================================
# REST API Interface (Optional)
# ==============================================================================

from typing import FastAPI
import uvicorn

class AuctionAPI:
    """REST API interface for the auction system"""
    
    def __init__(self, auction_agent: AuctionAgent, analytics: AuctionAnalytics):
        self.app = FastAPI(title="MemGPT Auction System API")
        self.auction_agent = auction_agent
        self.analytics = analytics
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/auctions/start")
        async def start_auction(service_data: dict):
            """Start a new auction"""
            try:
                service = Service(**service_data)
                result = await self.auction_agent.start_auction(service)
                self.analytics.record_auction_result(result)
                return {"status": "success", "result": result}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/auctions/analytics")
        async def get_analytics():
            """Get auction analytics"""
            return {
                "agent_performance": self.analytics.agent_performance,
                "strategy_performance": self.analytics.get_strategy_performance(),
                "total_auctions": len(self.analytics.auction_history)
            }
        
        @self.app.post("/auctions/strategy")
        async def change_strategy(strategy_data: dict):
            """Change selection strategy"""
            strategy_name = strategy_data.get("strategy_name")
            config = AuctionSystemConfig()
            strategy = config.get_strategy(strategy_name)
            self.auction_agent.set_selection_strategy(strategy)
            return {"status": "success", "strategy": strategy.get_strategy_name()}
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered agents"""
            return {
                "auction_agent": self.auction_agent.agent_id,
                "bidding_agents": self.auction_agent.bidding_agents
            }

# ==============================================================================
# Main Execution with Enhanced Features
# ==============================================================================

async def run_enhanced_auction_demo():
    """Run enhanced demonstration with learning agents and analytics"""
    logger.info("🚀 Starting Enhanced MemGPT Multi-Agent Auction System")
    
    # Initialize enhanced system
    comm_protocol = A2ACommunicationProtocol()
    analytics = AuctionAnalytics()
    config = AuctionSystemConfig()
    
    # Create auction agent
    auction_agent = AuctionAgent(
        agent_id="enhanced_auction_coordinator",
        llm_config=create_llm_config(),
        communication_protocol=comm_protocol,
        selection_strategy=config.get_strategy('weighted_score')
    )
    await auction_agent.initialize()
    
    # Create enhanced bidding agents
    enhanced_agents = []
    
    for i, (name, specialties, rate) in enumerate([
        ("adaptive_web_dev", [ServiceType.WEB_DEVELOPMENT], 80.0),
        ("learning_data_analyst", [ServiceType.DATA_ANALYSIS], 90.0),
        ("smart_content_writer", [ServiceType.CONTENT_WRITING], 55.0),
        ("versatile_generalist", list(ServiceType), 65.0)
    ]):
        agent = EnhancedBiddingAgent(
            agent_id=name,
            llm_config=create_llm_config(),
            communication_protocol=comm_protocol,
            specialties=specialties,
            base_rate=rate
        )
        agent.risk_tolerance = 0.3 + (i * 0.2)  # Vary risk tolerance
        await agent.initialize()
        enhanced_agents.append(agent)
        auction_agent.register_bidding_agent(agent.agent_id)
    
    # Run multiple auction rounds to demonstrate learning
    services = [
        Service("web_001", ServiceType.WEB_DEVELOPMENT, "E-commerce platform", {}, "2024-01-15", (2000, 5000)),
        Service("data_002", ServiceType.DATA_ANALYSIS, "Customer analytics", {}, "2024-01-20", (1500, 3500)),
        Service("content_003", ServiceType.CONTENT_WRITING, "Blog content", {}, "2024-01-10", (800, 1500)),
        Service("web_004", ServiceType.WEB_DEVELOPMENT, "Mobile app backend", {}, "2024-01-25", (3000, 6000)),
        Service("data_005", ServiceType.DATA_ANALYSIS, "Predictive modeling", {}, "2024-01-30", (2000, 4000)),
    ]
    
    # Test different strategies across rounds
    strategies_to_test = ['lowest_price', 'value_based', 'weighted_score']
    
    logger.info("Running multiple auction rounds to demonstrate learning...")
    
    for round_num, service in enumerate(services, 1):
        logger.info(f"\n🔄 ROUND {round_num}: {service.description}")
        
        # Rotate strategies
        strategy_name = strategies_to_test[round_num % len(strategies_to_test)]
        auction_agent.set_selection_strategy(config.get_strategy(strategy_name))
        
        # Run auction
        result = await auction_agent.start_auction(service, auction_duration_seconds=8)
        analytics.record_auction_result(result)
        
        # Show learning progress
        logger.info(f"Winner: {result['winner_id']} | Price: ${result['winning_price']} | Strategy: {strategy_name}")
        
        # Display agent win rates after each round
        for agent in enhanced_agents:
            win_rate = analytics.get_agent_win_rate(agent.agent_id)
            logger.info(f"  {agent.agent_id}: {win_rate:.1%} win rate")
        
        await asyncio.sleep(1)
    
    # Final analytics report
    logger.info("\n📊 FINAL ANALYTICS REPORT")
    logger.info("=" * 50)
    
    strategy_performance = analytics.get_strategy_performance()
    for strategy, stats in strategy_performance.items():
        logger.info(f"\n{strategy.upper()} Strategy:")
        logger.info(f"  Auctions: {stats['total_auctions']}")
        logger.info(f"  Avg Price: ${stats['avg_winning_price']:.2f}")
        logger.info(f"  Avg Bids per Auction: {stats['avg_bids_per_auction']:.1f}")
    
    logger.info(f"\nAgent Performance:")
    for agent_id, performance in analytics.agent_performance.items():
        win_rate = performance['wins'] / max(performance['total_bids'], 1)
        logger.info(f"  {agent_id}: {win_rate:.1%} win rate, {performance['wins']} wins, ${performance['avg_winning_price']:.2f} avg")
    
    return auction_agent, enhanced_agents, analytics

if __name__ == "__main__":
    # Choose which demo to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "enhanced":
        asyncio.run(run_enhanced_auction_demo())
    else:
        asyncio.run(run_auction_demo())