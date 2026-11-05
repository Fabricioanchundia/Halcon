"""Catálogos fijos del simulador: ciudades de Ecuador y categorías de comercio.

Los pesos de las ciudades son aproximados (proporcionales a su población) y solo sirven
para repartir clientes y comercios de forma creíble. Ecuador usa dólares, así que todos
los montos están en USD.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    weight: float


@dataclass(frozen=True)
class Category:
    name: str
    mcc: str  # Merchant Category Code (código ISO 18245, aproximado)
    channel: str  # "pos" (tarjeta presente) u "online" (tarjeta no presente)
    amount_multiplier: float
    merchant_weight: float


CITIES: tuple[City, ...] = (
    City("Guayaquil", -2.19, -79.89, 2.70),
    City("Quito", -0.18, -78.47, 2.00),
    City("Cuenca", -2.90, -79.00, 0.60),
    City("Santo Domingo", -0.25, -79.17, 0.45),
    City("Ambato", -1.24, -78.62, 0.40),
    City("Portoviejo", -1.05, -80.45, 0.30),
    City("Machala", -3.26, -79.96, 0.30),
    City("Manta", -0.95, -80.73, 0.27),
    City("Riobamba", -1.67, -78.65, 0.25),
    City("Loja", -4.00, -79.20, 0.20),
    City("Esmeraldas", 0.97, -79.65, 0.20),
    City("Ibarra", 0.35, -78.12, 0.20),
)

CATEGORIES: tuple[Category, ...] = (
    Category("groceries", "5411", "pos", 0.8, 0.25),
    Category("restaurants", "5812", "pos", 0.6, 0.15),
    Category("fuel", "5541", "pos", 0.9, 0.12),
    Category("pharmacy", "5912", "pos", 0.5, 0.08),
    Category("clothing", "5651", "pos", 1.5, 0.08),
    Category("electronics", "5732", "pos", 4.0, 0.03),
    Category("online_retail", "5964", "online", 1.5, 0.12),
    Category("digital_goods", "5816", "online", 0.4, 0.08),
    Category("travel", "4722", "online", 6.0, 0.02),
    Category("food_delivery", "5814", "online", 0.7, 0.07),
)

ONLINE_CITY = "ONLINE"
